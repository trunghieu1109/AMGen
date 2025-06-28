async def forward_180(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1a = (
        "Sub-task 1a: Collect and tabulate quantitative data on total solar neutrino fluxes (in cm^-2 s^-1) "
        "and energy spectra for all relevant branches: pp-I, pp-II (including Be-7 lines at 0.862 MeV and 0.384 MeV), "
        "pp-III (B-8 continuous spectrum), pep, and CNO. Provide numerical flux values and spectral characteristics."
    )
    N = self.max_sc
    cot_agents_1a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1a = []
    thinkingmapping_1a = {}
    answermapping_1a = {}
    subtask_desc1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_sc_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1a, answer1a = await cot_agents_1a[i]([taskInfo], cot_sc_instruction_1a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1a[i].id}, collecting quantitative flux data, thinking: {thinking1a.content}; answer: {answer1a.content}")
        possible_answers_1a.append(answer1a.content)
        thinkingmapping_1a[answer1a.content] = thinking1a
        answermapping_1a[answer1a.content] = answer1a
    most_common_answer_1a = Counter(possible_answers_1a).most_common(1)[0][0]
    thinking1a = thinkingmapping_1a[most_common_answer_1a]
    answer1a = answermapping_1a[most_common_answer_1a]
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    subtask_desc1a['response'] = {"thinking": thinking1a, "answer": answer1a}
    logs.append(subtask_desc1a)
    print("Step 1a: ", sub_tasks[-1])

    cot_sc_instruction_1b = (
        "Sub-task 1b: Analyze and summarize the spectral energy distributions of each neutrino branch, "
        "identifying energy ranges and spectral shapes, including monoenergetic lines (e.g., Be-7 at 0.862 MeV) "
        "and continuous spectra (e.g., B-8), to determine which branches contribute neutrinos within the 700-800 keV and 800-900 keV bands."
    )
    cot_agents_1b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1b = []
    thinkingmapping_1b = {}
    answermapping_1b = {}
    subtask_desc1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_sc_instruction_1b,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1b, answer1b = await cot_agents_1b[i]([taskInfo, thinking1a, answer1a], cot_sc_instruction_1b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1b[i].id}, analyzing spectral distributions, thinking: {thinking1b.content}; answer: {answer1b.content}")
        possible_answers_1b.append(answer1b.content)
        thinkingmapping_1b[answer1b.content] = thinking1b
        answermapping_1b[answer1b.content] = answer1b
    most_common_answer_1b = Counter(possible_answers_1b).most_common(1)[0][0]
    thinking1b = thinkingmapping_1b[most_common_answer_1b]
    answer1b = answermapping_1b[most_common_answer_1b]
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    subtask_desc1b['response'] = {"thinking": thinking1b, "answer": answer1b}
    logs.append(subtask_desc1b)
    print("Step 1b: ", sub_tasks[-1])

    debate_instruction_2a = (
        "Sub-task 2a: Quantitatively estimate the neutrino flux contributions from each solar neutrino branch within the 700-800 keV energy band under normal solar conditions, "
        "using spectral data and total fluxes from stage 1. Provide numerical flux values and justifications."
    )
    debate_agents_2a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2a = self.max_round
    all_thinking_2a = [[] for _ in range(N_max_2a)]
    all_answer_2a = [[] for _ in range(N_max_2a)]
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": debate_instruction_2a,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2a):
        for i, agent in enumerate(debate_agents_2a):
            if r == 0:
                thinking2a, answer2a = await agent([taskInfo, thinking1b, answer1b], debate_instruction_2a, r, is_sub_task=True)
            else:
                input_infos_2a = [taskInfo, thinking1b, answer1b] + all_thinking_2a[r-1] + all_answer_2a[r-1]
                thinking2a, answer2a = await agent(input_infos_2a, debate_instruction_2a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, estimating flux in 700-800 keV band, thinking: {thinking2a.content}; answer: {answer2a.content}")
            all_thinking_2a[r].append(thinking2a)
            all_answer_2a[r].append(answer2a)
    final_decision_agent_2a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2a, answer2a = await final_decision_agent_2a([taskInfo] + all_thinking_2a[-1] + all_answer_2a[-1], "Sub-task 2a: Make final decision on flux contributions in 700-800 keV band.", is_sub_task=True)
    agents.append(f"Final Decision agent 2a, flux estimation 700-800 keV, thinking: {thinking2a.content}; answer: {answer2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])

    debate_instruction_2b = (
        "Sub-task 2b: Quantitatively estimate the neutrino flux contributions from each solar neutrino branch within the 800-900 keV energy band under normal solar conditions, "
        "using spectral data and total fluxes from stage 1. Provide numerical flux values and justifications."
    )
    debate_agents_2b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2b = self.max_round
    all_thinking_2b = [[] for _ in range(N_max_2b)]
    all_answer_2b = [[] for _ in range(N_max_2b)]
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": debate_instruction_2b,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2b):
        for i, agent in enumerate(debate_agents_2b):
            if r == 0:
                thinking2b, answer2b = await agent([taskInfo, thinking1b, answer1b], debate_instruction_2b, r, is_sub_task=True)
            else:
                input_infos_2b = [taskInfo, thinking1b, answer1b] + all_thinking_2b[r-1] + all_answer_2b[r-1]
                thinking2b, answer2b = await agent(input_infos_2b, debate_instruction_2b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, estimating flux in 800-900 keV band, thinking: {thinking2b.content}; answer: {answer2b.content}")
            all_thinking_2b[r].append(thinking2b)
            all_answer_2b[r].append(answer2b)
    final_decision_agent_2b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2b, answer2b = await final_decision_agent_2b([taskInfo] + all_thinking_2b[-1] + all_answer_2b[-1], "Sub-task 2b: Make final decision on flux contributions in 800-900 keV band.", is_sub_task=True)
    agents.append(f"Final Decision agent 2b, flux estimation 800-900 keV, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])

    cot_reflect_instruction_2c = (
        "Sub-task 2c: Cross-validate and critically evaluate the flux estimates for both energy bands by comparing with authoritative solar neutrino data or literature, "
        "ensuring physical plausibility and correct branch-to-band mapping. Correct any inconsistencies or assumptions."
    )
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2c = self.max_round
    cot_inputs_2c = [taskInfo, thinking2a, answer2a, thinking2b, answer2b]
    subtask_desc2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_reflect_instruction_2c,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a", "thinking of subtask_2b", "answer of subtask_2b"],
        "agent_collaboration": "Reflexion"
    }
    thinking2c, answer2c = await cot_agent_2c(cot_inputs_2c, cot_reflect_instruction_2c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2c.id}, cross-validating flux estimates, thinking: {thinking2c.content}; answer: {answer2c.content}")
    for i in range(N_max_2c):
        feedback, correct = await critic_agent_2c([taskInfo, thinking2c, answer2c], "please review the flux estimates cross-validation and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2c.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2c.extend([thinking2c, answer2c, feedback])
        thinking2c, answer2c = await cot_agent_2c(cot_inputs_2c, cot_reflect_instruction_2c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2c.id}, refining cross-validation, thinking: {thinking2c.content}; answer: {answer2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking2c.content}; answer - {answer2c.content}")
    subtask_desc2c['response'] = {"thinking": thinking2c, "answer": answer2c}
    logs.append(subtask_desc2c)
    print("Step 2c: ", sub_tasks[-1])

    cot_reflect_instruction_3 = (
        "Sub-task 3: Model the effect of hypothetically stopping the pp-III (B-8) branch about 8.5 minutes ago on the neutrino flux reaching Earth, "
        "considering neutrino travel time and assuming all other branches remain unchanged. Adjust the flux contributions in both energy bands accordingly."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2c, answer2c]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask_2c", "answer of subtask_2c"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, modeling pp-III stop effect, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the modeling of pp-III branch stopping effect and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining pp-III stop effect model, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Calculate the adjusted total neutrino fluxes in the 700-800 keV and 800-900 keV bands after the pp-III branch stops, "
        "summing contributions from all remaining branches based on the adjusted fluxes from subtask_3. Provide numerical results."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, calculating adjusted fluxes, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[most_common_answer_4]
    answer4 = answermapping_4[most_common_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Compute the approximate ratio of fluxes: flux(band 1, 700-800 keV) divided by flux(band 2, 800-900 keV) "
        "using the adjusted flux values from subtask_4. Provide numerical ratio and reasoning."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing flux ratio, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking_5[r].append(thinking5)
            all_answer_5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Make final decision on flux ratio.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating flux ratio, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = (
        "Sub-task 6: Compare the computed flux ratio to the provided multiple-choice options (0.1, 10, 1, 0.01) "
        "and select the correct answer choice (A, B, C, or D). Provide justification."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask_5", "answer of subtask_5"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting final answer choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
