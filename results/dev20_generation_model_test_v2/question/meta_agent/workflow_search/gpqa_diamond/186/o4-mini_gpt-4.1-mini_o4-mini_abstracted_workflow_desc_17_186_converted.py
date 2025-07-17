async def forward_186(self, taskInfo):
    from collections import Counter
    print("Task Requirement:", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Gather stellar data (SC-CoT)
    cot_sc_instruction1 = (
        "Sub-task 1: Gather and summarize the input stellar data for all six targets, "
        "including RA, DEC, apparent V magnitude for Canopus and Polaris, and absolute V magnitude "
        "plus distances for the four M_V=15 stars. Record any assumptions."
    )
    N1 = self.max_sc
    cot_sc_agents1 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N1)
    ]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for agent in cot_sc_agents1:
        thinking1, answer1 = await agent([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, summarizing stellar data, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings1.append(thinking1)
        possible_answers1.append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1(
        [taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent stellar data summary from all analyses.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Extract instrument parameters (Debate)
    debate_instr2 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction2 = (
        "Sub-task 2: Extract ESPRESSO instrument parameters and performance from the official documentation or exposure-time calculator: "
        "telescope collecting area, total throughput, detector quantum efficiency, spectral resolution, bin width, and published S/N vs. V-mag curve."
        + debate_instr2
    )
    debate_agents2 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": debate_instruction2, "context": ["user query", thinking1, answer1], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents2:
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction2, r, is_sub_task=True)
            else:
                thinking2, answer2 = await agent(
                    [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1],
                    debate_instruction2, r, is_sub_task=True
                )
            agents.append(f"Debate agent {agent.id}, round {r}, extracting instrument parameters, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: Extract ESPRESSO instrument parameters. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Compute apparent V magnitudes (SC-CoT)
    cot_sc_instruction3 = (
        "Sub-task 3: Compute the apparent V magnitude of each of the four M_V=15 stars using the distance modulus (m_V = M_V + 5 log10(d/10 pc)). "
        "Explicitly note that differing distances produce differing m_V."
    )
    N3 = self.max_sc
    cot_sc_agents3 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N3)
    ]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query", thinking2, answer2], "agent_collaboration": "SC_CoT"}
    for agent in cot_sc_agents3:
        thinking3, answer3 = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, computing magnitudes, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings3.append(thinking3)
        possible_answers3.append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Synthesize and choose the most consistent apparent magnitudes for the M_V=15 stars.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Check sky visibility (SC-CoT)
    cot_sc_instruction4 = (
        "Sub-task 4: Check sky visibility at Paranal Observatory (latitude ≃ –24°): for each star’s declination, compute its maximum elevation. "
        "Exclude any star that never rises above the horizon."
    )
    N4 = self.max_sc
    cot_sc_agents4 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N4)
    ]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4, "context": ["user query", thinking3, answer3], "agent_collaboration": "SC_CoT"}
    for agent in cot_sc_agents4:
        thinking4, answer4 = await agent([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, checking visibility, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_thinkings4.append(thinking4)
        possible_answers4.append(answer4)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4,
        "Sub-task 4: Synthesize and choose the final list of sky-visible stars.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Calculate S/N per pixel (Debate)
    debate_instr5 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction5 = (
        "Sub-task 5: For each star that passes the sky-visibility check, calculate the expected S/N per binned pixel in a 1 h exposure using the instrument parameters from Subtask 2 "
        "and the star’s apparent magnitude from Subtask 3 (or known m_V). Avoid simplistic thresholding by computing S/N = sqrt(flux × time × area × throughput / noise) with all relevant terms (photon noise, read noise, sky background)."
        + debate_instr5
    )
    debate_agents5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", thinking4, answer4], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents5:
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                thinking5, answer5 = await agent(
                    [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1],
                    debate_instruction5, r, is_sub_task=True
                )
            agents.append(f"Debate agent {agent.id}, round {r}, computing S/N, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5(
        [taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Calculate S/N per pixel. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Label detectability (CoT)
    cot_instruction6 = (
        "Sub-task 6: Apply the detectability criterion S/N ≥ 10 to each star’s computed S/N. "
        "Clearly label which stars meet both the S/N threshold and sky-visibility requirement."
    )
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_instruction6, "context": ["user query", thinking5, answer5], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, labeling detectability, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Count detectable stars (CoT)
    cot_instruction7 = (
        "Sub-task 7: Count the total number of stars deemed detectable from Subtask 6 and select the matching answer choice among {2, 3, 4, 5}."
    )
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_instruction7, "context": ["user query", thinking6, answer6], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, counting detectable stars, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs