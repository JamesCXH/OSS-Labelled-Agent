from utils.element_utils.element_similarity import element_similarity
from pathlib import Path
import os
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image, ImageDraw
# import cv2
import copy as cp
import pickle
import re
from models import PageObservation
from playwright.sync_api import sync_playwright
from inferenceagent import *
import time
# from scrape7 import setup_context
from utils.page_interaction import setup_context
from utils import *
# from utils.inference_data import *  # here are the dataclasses for this script
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from utils.element_utils.element_similarity import element_similarity
from typing import List
import asyncio 

def load_scraper_state(file_path: str):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def match_tree_to_scrape(page, equiv_class_set):
    curr_page_html = page.content()
    equiv_class_set.get_class(page.url, curr_page_html)

def match_action_effects(curr_page_state: PageState, url_state_manager: URLStateManager) -> InferencePageState:  # Needless amounts of unrolling and rerolling
    found_state = url_state_manager.get_state(curr_page_state)
    if found_state:
        curr_page_actions = [action for action in curr_page_state.actions]  # (typeList, action)
        new_action_list = found_state.match_actions(curr_page_actions)
        inference_page_state = InferencePageState(curr_page_state.url, curr_page_state.ax_nodes, curr_page_state.html, found_state, new_action_list, curr_page_state.header_html, curr_page_state.footer_html, True)
        return inference_page_state

    else:
        print("URL STATE NOT FOUND FOR PAGE")
        curr_page_actions = [InferenceAction(action, None) for action in curr_page_state.actions]
        inference_page_state = InferencePageState(curr_page_state.url, curr_page_state.ax_nodes, curr_page_state.html,
                                                  None, curr_page_actions, curr_page_state.header_html,
                                                  curr_page_state.footer_html, False)
        return inference_page_state

async def get_chosen_element(page, chosen_indefinite, role_name_backup = True):
    chosen_action = chosen_indefinite.action
    if chosen_action.action_type is not None:
        type_list = [chosen_action.action_type]
        for item in chosen_indefinite.type_list:
            if item not in type_list:
                type_list.append(item)
    else:
        type_list = chosen_indefinite.type_list

    chosen_element = await get_element(page, chosen_action.xpath)
    # assert(traj_action.friendly_xpath != None)

    chosen_xpath = chosen_action.xpath
    if not chosen_element or await chosen_element.count() < 1 or await chosen_element.evaluate(
            "element => element.outerHTML") != chosen_action.html:  # perhaps do a stripped check
        print('First attempt failed')
        chosen_element = await get_element(page, chosen_action.friendly_xpath)
        chosen_xpath = chosen_action.friendly_xpath
        if not chosen_element or await chosen_element.count() < 1 or await chosen_element.evaluate(
                "element => element.outerHTML") != chosen_action.html:
            print('Second attempt failed')
            # now we try getting stuff at run time
            potentially_better_chosen_xpath = await get_xpath_by_outer_html(page, chosen_action.html)
            potentially_better_friendly_chosen_xpath = make_xpath_friendly(potentially_better_chosen_xpath)
            chosen_element = await get_element(page, potentially_better_friendly_chosen_xpath)
            chosen_xpath = potentially_better_friendly_chosen_xpath
            if not chosen_element or await chosen_element.count() < 1:
                print('Third attempt failed')
                chosen_element = await get_element(page, potentially_better_chosen_xpath)
                chosen_xpath = potentially_better_chosen_xpath
                if not chosen_element or await chosen_element.count() < 1:
                    print('Fourth attempt failed')
                    chosen_element = await get_element(page, chosen_action.friendly_xpath)
                    chosen_xpath = chosen_action.friendly_xpath
                    if not chosen_element or await chosen_element.count() < 1:
                        print('Fifth attempt failed')
                        chosen_element = await get_element(page, chosen_action.xpath)
                        chosen_xpath = chosen_action.xpath

    # If all XPath attempts failed, try locating by role and name
    if role_name_backup:
        print("ROLE NAME BACKUP ENABLED")
        print(f"BACKUP NAME: {chosen_action.name}")
        print(f"BACKUP ROLE: {chosen_action.role}")
        if chosen_element is None:
            print("CHOSEN ELEMENT IS NONE")
        else:
            print(f"CURRENT COUNT: {await chosen_element.count()}")
    print("GOT HERE INF HELP1")
    if (not chosen_element or (await chosen_element.count()) < 1) and chosen_action.name and chosen_action.role and role_name_backup:
    # if chosen_action.name and chosen_action.role and role_name_backup:
        print("GOT HERE INF HELP2")
        role, name = chosen_action.role, chosen_action.name
        print('All XPath attempts failed. Trying to locate by role and name.')
        elements = page.get_by_role(role, name=name)
        count = await elements.count()

        if count == 1:
            chosen_element = elements.first
            found_xpath = await xpath_from_element(chosen_element)
            chosen_xpath = make_xpath_friendly(found_xpath)  # MAY BE GOOD OR BAD, NEEDS MORE TESTING
            print(f'Element found by role and name: role="{role}", name="{name}"')
        elif count > 1:
            print(f'Fallback failed: Multiple elements found with role="{role}" and name="{name}".')
        else:
            print(f'Fallback failed: No elements found with role="{role}" and name="{name}".')

    if not chosen_element:
        print('Failed to locate the chosen element using all methods.')

    return chosen_element, chosen_xpath, type_list

def is_different_page(base_state, new_state):  # TODO, put this in some util after finalization
    # TODO JACCARD SIM THESE
    if len(base_state.actions) == 0 or len(new_state.actions) == 0:
        return False

    similar_count = 0
    for base_action in base_state.actions:
        base_html = base_action.action.html
        found = False
        for new_action in new_state.actions:
            if found:
                break
            else:
                new_html = new_action.action.html
                if element_similarity(base_html, new_html) >= 0.9:
                    similar_count += 1
                    found = True

    score = similar_count / (len(base_state.actions) + len(new_state.actions) - similar_count)
    if score >= 0.9:  # this was arbitrary, not enough empirical data
        return False
    return True


async def do_action_flow(page, chosen_action, chosen_element, chosen_xpath, type_list):
    chosen_action_screenshot, screenshot_success = await take_screenshot(
        page)  # we take a screenshot in case there's nothing to scroll to



    success = await apply_action(page, chosen_action, chosen_action_screenshot, chosen_element, chosen_xpath,
                           type_list)

    return success