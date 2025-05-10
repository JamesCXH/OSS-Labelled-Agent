import copy
import asyncio
import gc
import time
from utils.inference_helpers import *
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
import base64
from playwright.async_api import async_playwright
import datetime
import os
import pickle
from utils.inference_data import *
from utils.trajectory_saves import *
from inferenceagent import (
    call_action_agent,
    call_unified_task_clarifier,
    call_reflect_agent,
    call_memory_agent,
    call_task_separator,
    call_input_agent,
    call_unified_question_cleaner,
    call_action_part1,
    call_action_part2
)


class Agent:
    def __init__(self, fast_mode=False, retry_cap=10, reload_cap=1, call_retries=3):
        self.stop_event = asyncio.Event()  # Use asyncio.Event for async compatibility
        self.cleaned_up = asyncio.Event()
        self.playwright_lock = asyncio.Lock()
        self.fast_mode = fast_mode
        self.retry_cap = retry_cap
        self.reload_cap = reload_cap
        self.call_retries = call_retries
        # self.reset() UI resets agent, delete agent after done
        self.initialize_index()

        # def reset(self):
        self.scraper_state_file = 'dominos/scraper_state.pkl'
        self.url_state_manager = load_scraper_state(self.scraper_state_file)
        self.task = None
        self.task_notes = ''
        self.action_mem = []
        self.runtime_qa = []
        self.world_mem = ""
        self.questions = []
        self.question_answers = []
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.stop_event.clear()
        self.playwright = None
        self.browser = None
        self.browser_context = None
        self.page = None
        self.cdp_session = None
        self.curr_save_node = None
        self.failed_count = 0
        self.reload_count = 0
        self.saved_trajectory = SavedTrajectory()
        self.init_special_actions()

    async def ask_user(self, question):
        await self.output_queue.put(('question', question))
        while not self.stop_event.is_set():
            try:
                response = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)
                return response
            except asyncio.TimeoutError:
                continue
        raise InterruptedError("Agent stopped")

    def initialize_index(self):
        print("Creating...")
        start = time.time()
        with open('./data/factsNEW.txt', 'r') as f:
            document = f.read()
        nodes = [TextNode(text=chunk, id_=i) for (i, chunk) in enumerate(document.split('***'))]
        self.index = VectorStoreIndex(nodes)
        self.retriever = self.index.as_retriever(vector_store_query_mode="mmr",
                                                 vector_store_kwargs={"mmr_threshold": 1})
        print("took", time.time() - start)

    async def chained_action_call(self, curr_inf_tree, provider, model):
        start_time = time.time()
        print("CALLING CHAIN")
        summarized_info = await call_action_part1(self.task, self.task_notes, curr_inf_tree.get_raw_tree(), self.action_mem, self.item_context_pairs, provider=provider, model=model)
        print(summarized_info.parsed_output)
        print(f"ACTION CALL TIME first: {time.time() - start_time}")
        print("STEP 1 DONE")
        action_out_call = await call_action_part2(self.task, self.task_notes, summarized_info.parsed_output, curr_inf_tree)
        print(action_out_call.parsed_output)
        print("STEP 2 DONE")
        print(f"ACTION CALL TIME all: {time.time() - start_time}")
        return action_out_call


    async def formulate_questions(self):
        def autoregressive_retrieve(index, task, k=2):
            new_task = task
            nodes = []
            for i in range(k):
                retriever = index.as_retriever(similarity_top_k=i + 1)
                new_node = retriever.retrieve(new_task)[-1]
                new_task += new_node.get_content()
                nodes.append(new_node)
            return nodes

        top_k = 2
        self.context_info = "\n".join([node.get_content() for node in self.retriever.retrieve(self.task)])

        # Await the asynchronous call_task_separator
        agent_call = await call_task_separator(self.task)
        self.interesting_items = agent_call.parsed_output if agent_call.parsed_output else []

        self.item_context_pairs = "\n\n".join([
            "Item: " + item + "\n" + "Context: " + "".join(
                [node.get_content() for node in autoregressive_retrieve(self.index, item, top_k)]
            ) for item in self.interesting_items
        ])
        print("Item context pairs\n", self.item_context_pairs)

        self.saved_trajectory.important_info = self.item_context_pairs

        # Read questions.txt asynchronously
        def read_questions():
            with open('data/questions.txt', 'r') as f:
                return f.read()

        self.question_context = await asyncio.to_thread(read_questions)
        print("Question Context\n", self.question_context)

        # Await the asynchronous call_unified_task_clarifier
        agent_call = await call_unified_task_clarifier(self.task, self.question_context)
        self.questions = agent_call.parsed_output if agent_call.parsed_output else []

    async def launch_browser(self):
        async with self.playwright_lock:
            if self.playwright is None:
                self.playwright = await async_playwright().start()
            if self.browser is None:
                self.browser = await self.playwright.chromium.launch(headless=False)
            if self.browser_context is None or self.page is None or self.cdp_session is None:
                try:
                    self.browser_context, self.page, self.cdp_session, _ = await setup_context(self.browser, cookies=None, logged_in=False)
                except Exception as e:
                    print(f"Error during setup_context: {e}")
                    await self.cleanup_browser()
                    raise

    async def cleanup_browser(self):
        async with self.playwright_lock:
            try:
                if self.cdp_session:
                    await self.cdp_session.detach()
                if self.page:
                    await self.page.close()
                if self.browser_context:
                    await self.browser_context.close()
                if self.browser:
                    await self.browser.close()
                if self.playwright:
                    await self.playwright.stop()
            except Exception as e:
                print(f"Error during browser cleanup: {e}")
            finally:
                self.playwright = None
                self.browser = None
                self.browser_context = None
                self.page = None
                self.cdp_session = None
            gc.collect()
            self.cleaned_up.set()

    async def capture_and_send_screenshot(self, save_node=None):
        # async with self.playwright_lock:
        if self.page:
            screenshot = await self.page.screenshot(full_page=False)
            base64_screenshot = base64.b64encode(screenshot).decode('utf-8')
            if save_node is not None:
                save_node.screenshot = base64_screenshot
            await self.output_queue.put(('screenshot', base64_screenshot))

    async def check_if_loaded_screenshot(self):
        async with self.playwright_lock:
            if self.page:
                screenshot = await self.page.screenshot(full_page=False)
                base64_screenshot = base64.b64encode(screenshot).decode('utf-8')
                data_url = f"data:image/png;base64,{base64_screenshot}"
                await call_check_load_agent_screenshot(data_url)

    async def get_light_tree(self):
        ax_nodes = await get_ax_tree_no_extras(self.cdp_session)
        cleaned = AxObservation(ax_nodes, self.page.url, numbered=False)
        return cleaned

    def init_special_actions(self):
        stop_action = Action(Action.Type.STOP, None, None)
        stop_action.set_special_effect(
            'STOP')
        stop_indefinite = IndefiniteAction([Action.Type.STOP], stop_action, None,
                                           IndefiniteAction.Location.SPECIAL)

        input_all_action = Action(Action.Type.INPUT_GIVEN_INTENT, None, None)
        input_all_action.set_special_effect(
            'WRITE TEXT MODE')
        input_all_indefinite = IndefiniteAction([Action.Type.INPUT_GIVEN_INTENT], input_all_action,
                                                None,
                                                IndefiniteAction.Location.SPECIAL)

        ask_user_action = Action(Action.Type.REQUEST_USER_INPUT, None, None)
        ask_user_action.set_special_effect(
            'ASK USER')
        ask_user_indefinite = IndefiniteAction([Action.Type.REQUEST_USER_INPUT], ask_user_action,
                                                None,
                                                IndefiniteAction.Location.SPECIAL)

        reload_action = Action(Action.Type.RELOAD_PAGE, None, None)
        reload_action.set_special_effect(
            'RELOAD PAGE')
        reload_indefinite = IndefiniteAction([Action.Type.RELOAD_PAGE], reload_action,
                                               None,
                                               IndefiniteAction.Location.SPECIAL)

        self.special_actions = [stop_indefinite, input_all_indefinite, ask_user_indefinite, reload_indefinite]

    async def wait_for_network_idle(self, idle_time=0.2, timeout=1.0):
        """
        Wait until there are no network requests for `idle_time` seconds,
        but no longer than `timeout` seconds in total.

        Args:
            idle_time (float): Seconds of no network activity to consider idle.
            timeout (float): Maximum seconds to wait.

        Raises:
            asyncio.TimeoutError: If the network does not become idle within `timeout`.
        """
        active_requests = set()
        idle_event = asyncio.Event()

        async def set_idle():
            await asyncio.sleep(idle_time)
            if not active_requests:
                idle_event.set()

        def on_request(request):
            active_requests.add(request)
            idle_event.clear()

        def on_request_finished(request):
            active_requests.discard(request)
            if not active_requests:
                asyncio.create_task(set_idle())

        def on_request_failed(request):
            active_requests.discard(request)
            if not active_requests:
                asyncio.create_task(set_idle())

        # Attach event listeners
        self.page.on("request", on_request)
        self.page.on("requestfinished", on_request_finished)
        self.page.on("requestfailed", on_request_failed)

        try:
            # Initial check: if no active requests, start idle timer
            if not active_requests:
                asyncio.create_task(set_idle())

            # Wait for idle_event or timeout
            await asyncio.wait_for(idle_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Timeout: Network did not become idle within {timeout} seconds.")
        finally:
            # Remove event listeners to prevent memory leaks
            self.page.off("request", on_request)
            self.page.off("requestfinished", on_request_finished)
            self.page.off("requestfailed", on_request_failed)

    async def loop_until_loaded(self, wait_time=6):
        await asyncio.sleep(1)
        start_time = time.time()
        node_count = len(await get_ax_tree_no_extras(self.cdp_session))
        try:
            network_start = time.time()
            # await self.page.wait_for_load_state('networkidle', timeout=1000)
            await self.wait_for_network_idle(idle_time=0.2, timeout=1)
            print(f"network idle waited {time.time() - network_start}")
        except:
            print("network idle timed out")
            pass
        intermediate_start = time.time()
        while time.time() - start_time <= wait_time:
            new_node_count = len(await get_ax_tree_no_extras(self.cdp_session))
            if new_node_count == node_count and new_node_count > 0:
                print(f"skipped due to same node length after: {time.time() - intermediate_start}")
                break
            else:
                node_count = new_node_count
                await asyncio.sleep(0.2)

    async def run(self):
        await self.launch_browser()
        await self.page.goto('https://www.dominos.com/')
        try:
            await self.capture_and_send_screenshot()
            if not self.stop_event.is_set():
                if self.task is None:
                    self.task = await self.ask_user("What do you want done on dominos?")
                    self.saved_trajectory.user_input_task = self.task

            if not self.stop_event.is_set():
                await self.formulate_questions()
                print(self.questions)
                for question in self.questions:
                    if self.stop_event.is_set():
                        break
                    answer = await self.ask_user(question)
                    self.question_answers.append((question, answer))
                    # Check stop_event after each user response
                    if self.stop_event.is_set():
                        break

            self.saved_trajectory.question_answers = self.question_answers

            if not self.stop_event.is_set():
                agent_call = await call_unified_question_cleaner(self.task, self.question_answers)
                if agent_call.parsed_output:
                    self.task = agent_call.parsed_output
                    notes_out_call = await call_unified_context_cleaner(self.task, self.task_notes, self.context_info)
                    self.task_notes = notes_out_call.parsed_output
                    # input("*** Updated Task Notes ***" + self.task_notes)
                    # self.cleaned_task = self.saved_trajectory.cleaned_task

            matched_inference_state = None
            curr_page_state = None
            curr_inf_tree = None

            # Main action loop
            while not self.stop_event.is_set():
                updated_page_state = False
                try:
                    self.curr_save_node = SavedTrajectoryNode()

                    await self.capture_and_send_screenshot(self.curr_save_node)


                    async with self.playwright_lock:



                        self.curr_save_node.url = self.page.url


                        if not matched_inference_state and not curr_page_state:  # curr_page_state should always be defined unless it's the first one, so if inference state isn't matched script should be breaking as intended
                            curr_page_state = await get_page_state(self.page, self.cdp_session)
                            matched_inference_state = match_action_effects(curr_page_state, self.url_state_manager)
                            updated_page_state = True

                        if not curr_inf_tree:  # should only be used during first iteration
                            curr_inf_tree = InferenceAxtree(matched_inference_state, special_actions=self.special_actions,
                                                            use_scrape=True, url=self.page.url)

                        if self.stop_event.is_set():
                            break



                        start = time.time()
                        # Check stop_event after API call
                        if self.stop_event.is_set():
                            break

                        action_out_call = await self.chained_action_call(curr_inf_tree, "anthropic", model="claude-3-5-sonnet-latest")

                        if action_out_call.parsed_output is None:
                            self.failed_count += 1
                            break

                        action_out_list = [action_out_call.parsed_output]  # a single action


                        self.curr_save_node.ad_call = action_out_call
                        if self.stop_event.is_set():
                            break


                        print("Action took", time.time() - start)
                        print(action_out_list)

                        if action_out_list is None:
                            # Now as action_out is a list, this may not ever be None due to structured outputs
                            print("NO ACTION LIST")
                            raise Exception

                        if action_out_list == []:
                            print("NO ACTIONS GIVEN")

                        if action_out_list == [None]:
                            print("THIS SHOULD BE IMPOSSIBLE, CAN'T HAVE NONE IN LIST")

                        old_inf_tree = copy.deepcopy(curr_inf_tree)  # do we need to copy this?

                        for (iterating_action_index, action_out) in enumerate(action_out_list):
                            new_reflect_action_indices = []
                            intermediate_tree = None
                            something_failed = False

                            chosen_action_index, reason_for_action = action_out  # we choose a list of actions in support, everything set up like input
                            reason_for_action = reason_for_action.encode('utf-8').decode('unicode_escape')

                            chosen_indefinite = old_inf_tree.get_action_from_index(chosen_action_index)
                            chosen_action = chosen_indefinite.action

                            #  HARD STOPS FOR DOMINO'S

                            if chosen_action is not None and chosen_action.html is not None and (
                                    'payment-order-now' in chosen_action.html or 'Place Your Order' in chosen_action.html):
                                await self.output_queue.put(
                                    ('exit_message', "Stopping agent to prevent actually buying a Pizza"))
                                self.stop()
                            elif chosen_indefinite.location != IndefiniteAction.Location.SPECIAL and "pages/order/payment" in self.page.url:  # may be a bad check
                                await self.output_queue.put(
                                    ('exit_message', "Stopping agent to prevent actually buying a Pizza"))
                                self.stop()

                            if self.stop_event.is_set():
                                break


                            """
                            
                            As a matter of philosophy, if we fail in performing an action we don't tell reflect that.
                            It's not the bot messing up, it's not the website being bad, it's us.
                            
                            """


                            if chosen_indefinite.location != IndefiniteAction.Location.SPECIAL:
                                """
                                
                                This generally does an action
                                
                                """

                                use_role_name_backup = curr_page_state.all_tree_lines.count(chosen_action.tree_line) == 1 and chosen_action.tree_line != ""

                                chosen_element, chosen_xpath, type_list = await get_chosen_element(
                                    self.page,
                                    chosen_indefinite,
                                    use_role_name_backup
                                )

                                success = await do_action_flow(
                                    self.page, chosen_action, chosen_element, chosen_xpath,
                                    type_list
                                )

                                if success:
                                    await self.output_queue.put(('only_out', reason_for_action))
                                    new_reflect_action_indices.append(chosen_action_index)
                                else:
                                    something_failed = True

                            elif chosen_action.action_type == Action.Type.RELOAD_PAGE:
                                self.reload_count += 1
                                if self.reload_count > self.reload_cap:
                                    await self.page.reload()
                                    self.reload_count = 0


                            elif chosen_action.action_type == Action.Type.REQUEST_USER_INPUT:
                                """
                                
                                Have some flag to stop reflecting on this, fix the current reflect pruning method.
                                Reason about this harder.
                                
                                Then call an agent, given curr axtree and questions, formulate questions which we ask the user.
                                
                                Shove all of these in actionmem. action mem now needs to be more sophisticated, can't remove these QAs from the memory.
                                
                                
                                TODO, SUPPORT SKIPPING OVER ACTIONS WHERE THE LAST ACTION WAS/WASN'T ASKING QUESTIONS, QUESTIONS MAY BE INTERMEDIATE.ETC
                                """
                                question_out_call = await call_intermediate_questions_agent(self.task, old_inf_tree.get_question_tree(), reason_for_action, self.task_notes)
                                intermediate_questions = question_out_call.parsed_output

                                question_string = ''

                                for question in intermediate_questions:
                                    if self.stop_event.is_set():
                                        break
                                    answer = await self.ask_user(question)

                                    question_string += f"Q: {question}\nA: {answer}\n"

                                    self.runtime_qa.append(f"Q: {question}\nA: {answer}")
                                    # Check stop_event after each user response
                                    if self.stop_event.is_set():
                                        break
                                if question_string:
                                    notes_out_call = await call_unified_notes_cleaner(self.task, self.task_notes, question_string)
                                    self.task_notes = notes_out_call.parsed_output
                                    # input("*** Updated Task Notes ***" + self.task_notes)
                                # TODO MAKE THIS INTO A MEMORY INJECTION AND ADD IT IN
                                new_memory = LinearMemory(location_details=question_string,
                                                          difference_reasoning="",
                                                          intent=reason_for_action,
                                                          action_treelines=old_inf_tree.get_action_treelines(
                                                              [chosen_action_index]),
                                                          page_url=old_inf_tree.url,
                                                          is_question=True)

                                self.action_mem.append(new_memory)



                            elif chosen_action.action_type == Action.Type.STOP:
                                await self.output_queue.put(('exit_message', "Agent has stopped as task is complete. "))
                                self.stop()

                            elif chosen_action.action_type == Action.Type.INPUT_GIVEN_INTENT:
                                new_reflect_action_indices.append(chosen_action_index)
                                await self.output_queue.put(('only_out', reason_for_action))
                                input_phase_count = 0

                                while True:
                                    seen_question = False
                                    question_string = ''
                                    if input_phase_count > 3:
                                        break
                                    desired = await call_input_agent(
                                        self.task, reason_for_action,
                                        old_inf_tree.get_input_tree(), self.context_info, self.task_notes
                                    )

                                    if desired.parsed_output is None or desired.parsed_output[0] is None or desired.parsed_output[0][0] is None:
                                        print("INPUT AGENT PARSED OUTPUT IS NONE")
                                        raise Exception

                                    for (chosen_input_index, chosen_text) in desired.parsed_output:
                                        if self.stop_event.is_set():
                                            break

                                        chosen_input_indefinite = old_inf_tree.get_action_from_index(
                                            chosen_input_index)
                                        chosen_input_action = chosen_input_indefinite.action

                                        if chosen_input_action.action_type == Action.Type.REQUEST_USER_INPUT:
                                            seen_question = True


                                            if self.stop_event.is_set():
                                                break

                                            answer = await self.ask_user(chosen_text)
                                            question_string += f"Q: {chosen_text}\nA: {answer}\n"

                                            self.runtime_qa.append(
                                                f"Q: {chosen_text}\nA: {answer}")

                                            if self.stop_event.is_set():
                                                break

                                        else:
                                            use_role_name_backup = curr_page_state.all_tree_lines.count(
                                                chosen_input_action.tree_line) == 1 and chosen_input_action.tree_line != ""

                                            chosen_element, chosen_xpath, type_list = await get_chosen_element(
                                                self.page,
                                                chosen_input_indefinite,
                                                use_role_name_backup
                                            )

                                            chosen_input_action.set_input_string(chosen_text)
                                            if self.stop_event.is_set():
                                                break

                                            success = await do_action_flow(
                                                self.page, chosen_input_action, chosen_element, chosen_xpath,
                                                type_list
                                            )

                                            if success:
                                                new_reflect_action_indices.append(chosen_input_index)
                                            else:
                                                break

                                            if self.stop_event.is_set():
                                                break
                                    if question_string:
                                        # print(question_string)
                                        notes_out_call = await call_unified_notes_cleaner(self.task, self.task_notes, question_string)
                                        self.task_notes = notes_out_call.parsed_output
                                        # input("*** Updated Task Notes ***" + self.task_notes)

                                    # TODO MAKE THIS INTO A MEMORY INJECTION AND ADD IT IN


                                    if not seen_question:
                                        break
                                    else:
                                        new_memory = LinearMemory(
                                            location_details=question_string,
                                            difference_reasoning="",
                                            intent=reason_for_action,
                                            action_treelines=old_inf_tree.get_action_treelines(
                                                [chosen_action_index]),
                                            page_url=old_inf_tree.url,
                                            is_question=True)

                                        self.action_mem.append(new_memory)

                                    input_phase_count += 1

                            await self.capture_and_send_screenshot(self.curr_save_node)

                            if iterating_action_index == len(
                                    action_out_list) - 1:  # if multiple actions, we assume only last action can possibly need load (THIS IS POTENTIALLY BAD)
                                #  standard wait for reflect, get full new tree
                                #  This is not the most elegant solution as it was patched when we had multi-action emission
                                start_time = time.time()
                                wait_time = 6
                                timeout_wait = wait_time + 1
                                try:
                                    await asyncio.wait_for(self.loop_until_loaded(wait_time=wait_time),
                                                           timeout=timeout_wait)
                                except asyncio.TimeoutError:
                                    print("Timeout reached while waiting for text to load.")

                                print(f"SUCCESS: Fast mode lapsed time: {time.time() - start_time}")


                            await self.capture_and_send_screenshot(self.curr_save_node)

                            curr_page_state = await get_page_state(self.page, self.cdp_session)

                            matched_inference_state = match_action_effects(curr_page_state,
                                                                           self.url_state_manager)

                            updated_page_state = True

                            curr_inf_tree = InferenceAxtree(matched_inference_state,
                                                            special_actions=self.special_actions,
                                                            use_scrape=True,
                                                            url=self.page.url)

                            intermediate_tree = curr_inf_tree.get_raw_tree()

                            if chosen_action.action_type != Action.Type.RELOAD_PAGE:
                                self.reload_count = 0

                            if something_failed:  # this is in action loop
                                """
                                
    
                                we remove the last item in action mem and just redo the reflect
                                reflect action indices should still be what was performed (successfully) in the call before this one where it failed,
                                old_inf_tree should still be what it was performed on
                                
                                WE NEED TO THINK HARDER ABOUT REDOING A REFLECT CALL, there is some not sufficiently loaded assumption failsafe
                                
                                """
                                self.failed_count += 1
                                if self.failed_count > self.retry_cap:
                                    self.stop()
                                break
                            else:
                                self.failed_count = 0

                                skip_types = [Action.Type.RELOAD_PAGE, Action.Type.STOP, Action.Type.REQUEST_USER_INPUT]

                                mem_response = ('', '', '')

                                if chosen_action.action_type not in skip_types: # add more
                                    new_reflect_action_indices = list(set(new_reflect_action_indices))
                                    successful_reflect = False
                                    for i in range(self.call_retries):
                                        reflect_response_call = await call_reflect_agent(
                                            reason_for_action,
                                            str(old_inf_tree.get_tree_with_specific_action_effect(new_reflect_action_indices)),
                                            intermediate_tree, self.task
                                        )


                                        mem_response = reflect_response_call.parsed_output
                                        if mem_response is None:
                                            print("NO MEM RESPONSE")
                                            continue

                                        if mem_response != ('', ''):
                                            successful_reflect = True
                                            break

                                    object_and_effect, difference_reasoning = mem_response[0], mem_response[1]
                                    if not successful_reflect:
                                        object_and_effect, difference_reasoning = 'N/A', 'N/A'


                                    self.curr_save_node.reflect_call.append(copy.deepcopy(reflect_response_call))
                                    print(mem_response)
                                    print(len(mem_response))

                                    new_memory = LinearMemory(location_details=object_and_effect,
                                                              difference_reasoning=difference_reasoning,
                                                              intent=reason_for_action,
                                                              action_treelines=old_inf_tree.get_action_treelines(new_reflect_action_indices),
                                                              page_url=old_inf_tree.url,
                                                              is_question=False)
                                    self.action_mem.append(new_memory)


                    self.saved_trajectory.add_node(copy.deepcopy(self.curr_save_node))

                    if self.stop_event.is_set():
                        break
                    gc.collect()
                except Exception as e:
                    print(f"Agent encountered inner exception: {e}")
                finally:
                    if not updated_page_state:
                        matched_inference_state = match_action_effects(curr_page_state,
                                                                       self.url_state_manager)

                        curr_inf_tree = InferenceAxtree(matched_inference_state,
                                                        special_actions=self.special_actions,
                                                        use_scrape=True,
                                                        url=self.page.url)

                    if self.stop_event.is_set():  # only set in self.stop()
                        print("Stop event detected. Initiating cleanup...")
                        print("CLEANING")
                        await self.cleanup_browser()
                        print("FINISHED CLEANING")
                        print("DONE!")
                    else:
                        if not updated_page_state:
                            curr_page_state = await get_page_state(self.page, self.cdp_session)

                            matched_inference_state = match_action_effects(curr_page_state,
                                                                           self.url_state_manager)

                        self.failed_count += 1
                        if self.failed_count > self.retry_cap:
                            self.stop()





        except Exception as e:
            print(f"Agent encountered an exception: {e}")
        finally:
            if self.stop_event.is_set():  # only set in self.stop()
                print("Stop event detected. Initiating cleanup...")
                print("CLEANING")
                await self.cleanup_browser()
                print("FINISHED CLEANING")
                print("DONE!")
            else:
                await self.output_queue.put(('exit_message', "Agent crashed, please reset."))
                self.stop()
                await self.cleanup_browser()

    def stop(self):
        print("Stop method called")
        # Get the current date and time
        # now = datetime.datetime.now()
        # filename = f"{self.saved_trajectory.user_input_task}_{now.strftime('%Y%m%d_%H%M')}.pkl"
        #
        # save_directory = 'saved_trajectories'
        # if not os.path.exists(save_directory):
        #     os.makedirs(save_directory)
        #
        # full_path = os.path.join(save_directory, filename)
        # with open(full_path, 'wb') as file:
        #     pickle.dump(self.saved_trajectory, file)
        self.stop_event.set()
