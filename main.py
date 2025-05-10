# main.py
import socketio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
import asyncio
from UIAgent import Agent
import gc
import time
import os
from collections import defaultdict

# Initialize Socket.IO server with ASGI mode
# sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
# Initialize Socket.IO server with ASGI mode
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    ping_interval=30,    # Ping every 30 seconds
    ping_timeout=120     # Timeout after 120 seconds without pong
)
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Set up templates directory
templates = Jinja2Templates(directory="templates")


agents = {}
agents_lock = asyncio.Lock()

# Mapping of client session IDs to keys
client_keys = {}

# Mapping of keys to sets of connected sids
key_clients = defaultdict(set)


@app.get("/", response_class=HTMLResponse)
async def get(request: Request, key: str = None):
    return templates.TemplateResponse("index.html", {"request": request, "key": key})


@sio.event
async def connect(sid, environ):
    # Extract key from query string
    query_string = environ.get('QUERY_STRING', '')
    key = None
    for param in query_string.split('&'):
        if param.startswith('key='):
            key = param.split('=')[1]
            break
    await sio.enter_room(sid, key)
    client_keys[sid] = key
    key_clients[key].add(sid)
    await sio.emit('connection_response', {'status': 'connected'}, to=sid)
    async with agents_lock:
        if key in agents:
            # Agent is already running
            await sio.emit('agent_started', {'status': 'Agent already running'}, to=sid)
            # Send the current state to the newly connected client
            await sio.emit('agent_state', {
                'messages': agents[key]['messages'],
                'browserScreenshot': agents[key]['browserScreenshot'],
                'waiting_for_input': agents[key]['waiting_for_input']
            }, to=sid)
        else:
            pass


@sio.event
async def disconnect(sid):
    key = client_keys.pop(sid, None)
    if key:
        await sio.leave_room(sid, key)
        key_clients[key].discard(sid)
        if not key_clients[key]:
            await reset_agent_for_key(key, reason="no_active_connections")


@sio.event
async def start_agent(sid):
    key = client_keys.get(sid)
    if not key:
        await sio.emit('error', {'message': 'No key associated with this connection'}, to=sid)
        return
    async with agents_lock:
        if key not in agents:
            agent = Agent(fast_mode=False)
            agents[key] = {
                'agent': agent,
                'messages': [],
                'browserScreenshot': '',
                'waiting_for_input': False,
                'input_timeout_task': None
            }
            # Start the agent's run method as a background task
            agent_task = asyncio.create_task(run_agent(key))
            # Start the background loop to handle agent outputs
            background_task = asyncio.create_task(agent_loop(key))
            agents[key]['agent_task'] = agent_task
            agents[key]['background_task'] = background_task
            await sio.emit('agent_started', {'status': 'Agent started'}, to=key)
        else:
            await sio.emit('agent_already_running', {'status': 'Agent is already running'}, to=sid)


@sio.event
async def reset_agent(sid):
    key = client_keys.get(sid)
    if not key:
        await sio.emit('error', {'message': 'No key associated with this connection'}, to=sid)
        return
    await reset_agent_for_key(key, reason="user_reset", emit_to_sid=sid)


@sio.event
async def user_response(sid, data):
    key = client_keys.get(sid)
    if not key:
        await sio.emit('error', {'message': 'No key associated with this connection'}, to=sid)
        return
    response = data.get('response')
    if not response:
        await sio.emit('error', {'message': 'No response provided'}, to=sid)
        return
    async with agents_lock:
        agent_info = agents.get(key)
        if agent_info:
            if not agent_info['waiting_for_input']:
                await sio.emit('error', {'message': 'Agent is not waiting for input'}, to=sid)
                return
            agent = agent_info['agent']
            # Append user message to the agent's message list
            agent_info['messages'].append({'type': 'user', 'text': response})
            agent_info['waiting_for_input'] = False
            # Cancel the input timeout task if it exists
            if agent_info.get('input_timeout_task'):
                agent_info['input_timeout_task'].cancel()
                agent_info['input_timeout_task'] = None
            await agent.input_queue.put(response)
            # Broadcast the user message to all clients in the key's room except sender
            await sio.emit('user_message', {'text': response}, to=key, skip_sid=sid)
            # Emit 'input_disabled' to all clients in the room to disable input fields
            await sio.emit('input_disabled', to=key)
        else:
            await sio.emit('error', {'message': 'Agent is not running'}, to=sid)


async def run_agent(key):
    agent_info = agents.get(key)
    if not agent_info:
        print(f"{time.time()}: No agent found for key {key}")
        return
    agent = agent_info['agent']
    try:
        await agent.run()
    except asyncio.CancelledError:
        print(f"{time.time()}: Agent for key {key} was cancelled.")
    except Exception as e:
        print(f"{time.time()}: Agent for key {key} encountered an error: {e}")
    finally:
        print(f"{time.time()}: Agent run method for key {key} finished")
        await sio.emit('agent_stopped', to=key)
        print(f"{time.time()}: Agent task for key {key} finished")
        # Do not reset the agent when it stops itself
        # Only clean up internal state
        async with agents_lock:
            if key in agents:
                del agents[key]
                gc.collect()


async def agent_loop(key):
    agent_info = agents.get(key)
    if not agent_info:
        return
    agent = agent_info['agent']
    while not agent.stop_event.is_set():
        if not agent.output_queue.empty():
            output_type, data = await agent.output_queue.get()
            if output_type == 'screenshot':
                agent_info['browserScreenshot'] = data  # Update the current screenshot
                await sio.emit('browser_update', {'screenshot': data}, to=key)
            elif output_type == 'question':
                message = {'type': 'agent', 'text': data}
                agent_info['messages'].append(message)  # Append agent question to messages
                agent_info['waiting_for_input'] = True
                await sio.emit('agent_question', {'question': data}, to=key)
                # Start the 45-second input timeout
                agent_info['input_timeout_task'] = asyncio.create_task(input_timeout(key))
            elif output_type == 'only_out':
                message = {'type': 'only-out', 'text': data}
                agent_info['messages'].append(message)  # Append agent-only message to messages
                await sio.emit('agent_only_out', {'message': data}, to=key)
            elif output_type == 'exit_message':  # **New Handling for exit_message**
                message = {'type': 'exit', 'text': data}
                agent_info['messages'].append(message)  # Append exit message to messages
                await sio.emit('agent_exit_message', {'message': data}, to=key)
        await asyncio.sleep(0.1)

    # Handle remaining messages after agent stops
    start_time = time.time()
    while not agent.output_queue.empty():
        output_type, data = await agent.output_queue.get()
        if output_type == 'screenshot':
            agent_info['browserScreenshot'] = data  # Update the current screenshot
            await sio.emit('browser_update', {'screenshot': data}, to=key)
        elif output_type == 'question':
            message = {'type': 'agent', 'text': data}
            agent_info['messages'].append(message)  # Append agent question to messages
            agent_info['waiting_for_input'] = True
            await sio.emit('agent_question', {'question': data}, to=key)
            # Start the 45-second input timeout
            agent_info['input_timeout_task'] = asyncio.create_task(input_timeout(key))
        elif output_type == 'only_out':
            message = {'type': 'only-out', 'text': data}
            agent_info['messages'].append(message)  # Append agent-only message to messages
            await sio.emit('agent_only_out', {'message': data}, to=key)
        elif output_type == 'exit_message':  # **New Handling for exit_message**
            message = {'type': 'exit', 'text': data}
            agent_info['messages'].append(message)  # Append exit message to messages
            await sio.emit('agent_exit_message', {'message': data}, to=key)
        await asyncio.sleep(0.1)
        if (time.time() - start_time) >= 10:
            break


async def input_timeout(key):
    try:
        await asyncio.sleep(45)
        # Acquire the lock only to check the condition
        async with agents_lock:
            agent_info = agents.get(key)
            should_reset = agent_info and agent_info['waiting_for_input'] and key_clients[key]
        if should_reset:
            print(f"{time.time()}: Agent waiting for input timed out for key {key}")
            await reset_agent_for_key(key, reason="input_timeout")
    except asyncio.CancelledError:
        # Timeout was cancelled because user responded
        pass


async def reset_agent_for_key(key, reason="unknown", emit_to_sid=None):
    """
    Resets the agent associated with the given key.

    Args:
        key (str): The key associated with the agent.
        reason (str): The reason for resetting ('no_active_connections', 'user_reset', 'input_timeout').
        emit_to_sid (str): Specific SID to emit messages to, if any.
    """
    async with agents_lock:
        agent_info = agents.get(key)
        if not agent_info:
            # Agent already reset
            return
        agent = agent_info['agent']
        print(f"{time.time()}: Stopping agent for key {key} due to {reason}...")
        if agent and not agent.cleaned_up.is_set():
            agent.stop()
            try:
                await asyncio.wait_for(agent.cleaned_up.wait(), timeout=20)
                print(f"{time.time()}: Agent for key {key} cleanup completed.")
            except asyncio.TimeoutError:
                print(f"{time.time()}: Warning: Agent for key {key} did not clean up within timeout.")
        # Cancel the agent task
        agent_task = agent_info.get('agent_task')
        if agent_task:
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
        # Cancel the background task
        background_task = agent_info.get('background_task')
        if background_task:
            background_task.cancel()
            try:
                await background_task
            except asyncio.CancelledError:
                pass
        # Cancel the input timeout task if it exists
        input_timeout_task = agent_info.get('input_timeout_task')
        if input_timeout_task:
            input_timeout_task.cancel()
            agent_info['input_timeout_task'] = None
        # Clear the agent's state
        del agents[key]
        gc.collect()
        print(f"{time.time()}: Agent reset completed for key {key} due to {reason}.")

    # Emit appropriate messages based on the reason outside the lock
    if reason in ["no_active_connections", "input_timeout"]:
        if reason == "no_active_connections":
            status_message = "Agent reset due to no active connections."
        elif reason == "input_timeout":
            status_message = "Agent reset due to input timeout."
        await sio.emit('agent_reset', {
            'status': status_message,
            'reason': reason
        }, to=key)
    elif reason == "user_reset":
        if emit_to_sid:
            await sio.emit('agent_reset', {
                'status': 'Agent reset by user.',
                'reason': reason
            }, to=emit_to_sid)
        else:
            await sio.emit('agent_reset', {
                'status': 'Agent reset.',
                'reason': reason
            }, to=key)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:socket_app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
