async def forward_104(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Calculate the effective temperature of the spotted hemisphere by accounting for the 20% filling factor of spots that are 1000K cooler than the star's effective temperature (6000K). Use the formula: T_eff_spotted = (f * T_spot^4 + (1-f) * T_star^4)^(1/4), where f=0.2, T_spot=5000K, T_star=6000K."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculate effective temperature of spotted hemisphere, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Determine the flux (brightness) of the spotted hemisphere and the unspotted hemisphere using the effective temperatures and filling factor from Sub-task 1. Use Stefan-Boltzmann law: flux ~ T^4. Calculate flux_spotted and flux_unspotted accordingly."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculate fluxes of spotted and unspotted hemispheres, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3a = "Sub-task 3a: Calculate the flux difference between the spotted hemisphere and the unspotted hemisphere based on the fluxes computed in Sub-task 2."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, calculate flux difference between spotted and unspotted hemispheres, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {"thinking": thinking3a, "answer": answer3a}
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    
    cot_instruction_3b = "Sub-task 3b: Calculate the relative amplitude of the brightness variation caused by the spotted hemisphere normalized by the total stellar flux (both hemispheres combined). Use amplitude = (flux_unspotted_hemisphere - flux_spotted_hemisphere) / (flux_unspotted_hemisphere + flux_unspotted_hemisphere)."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "CoT"
    }
    thinking3b, answer3b = await cot_agent_3b([taskInfo, thinking3a, answer3a], cot_instruction_3b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3b.id}, calculate normalized brightness amplitude, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {"thinking": thinking3b, "answer": answer3b}
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    
    cot_reflect_instruction_3c = "Sub-task 3c: Conduct a reflexion and validation step to critically examine and verify the amplitude calculation from Sub-task 3b, ensuring the normalization and physical interpretation are correct and the amplitude is physically plausible (e.g., less than 1 and consistent with spot coverage)."
    cot_agent_3c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3c = self.max_round
    cot_inputs_3c = [taskInfo, thinking3b, answer3b]
    subtask_desc3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_reflect_instruction_3c,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Reflexion"
    }
    thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, reflexion on amplitude calculation, thinking: {thinking3c.content}; answer: {answer3c.content}")
    for i in range(N_max_3c):
        feedback3c, correct3c = await critic_agent_3c([taskInfo, thinking3c, answer3c], "Review the amplitude calculation and confirm its physical plausibility and normalization correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3c.id}, feedback: {feedback3c.content}; correct: {correct3c.content}")
        if correct3c.content == "True":
            break
        cot_inputs_3c.extend([thinking3c, answer3c, feedback3c])
        thinking3c, answer3c = await cot_agent_3c(cot_inputs_3c, cot_reflect_instruction_3c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3c.id}, refining amplitude calculation, thinking: {thinking3c.content}; answer: {answer3c.content}")
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
    subtask_desc3c['response'] = {"thinking": thinking3c, "answer": answer3c}
    logs.append(subtask_desc3c)
    print("Step 3c: ", sub_tasks[-1])
    
    debate_instruction_4a = "Sub-task 4a: Compute the radius ratio (Rpl/Rstar) of a hypothetical exoplanet that would produce the same amplitude signal in the star's light curve as the brightness variation caused by the spots, using the amplitude validated in Sub-task 3c and the transit depth formula amplitude = (Rpl/Rstar)^2."
    debate_agents_4a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4a = self.max_round
    all_thinking4a = [[] for _ in range(N_max_4a)]
    all_answer4a = [[] for _ in range(N_max_4a)]
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": debate_instruction_4a,
        "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4a):
        for i, agent in enumerate(debate_agents_4a):
            if r == 0:
                thinking4a, answer4a = await agent([taskInfo, thinking3c, answer3c], debate_instruction_4a, r, is_sub_task=True)
            else:
                input_infos_4a = [taskInfo, thinking3c, answer3c] + all_thinking4a[r-1] + all_answer4a[r-1]
                thinking4a, answer4a = await agent(input_infos_4a, debate_instruction_4a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, compute radius ratio, thinking: {thinking4a.content}; answer: {answer4a.content}")
            all_thinking4a[r].append(thinking4a)
            all_answer4a[r].append(answer4a)
    final_decision_agent_4a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4a, answer4a = await final_decision_agent_4a([taskInfo] + all_thinking4a[-1] + all_answer4a[-1], "Sub-task 4a: Make final decision on radius ratio Rpl/Rstar.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating radius ratio, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])
    
    cot_instruction_4b = "Sub-task 4b: Quantitatively compare the computed radius ratio from Sub-task 4a with each multiple-choice option by calculating the absolute difference, and select the option with the smallest difference as the final answer."
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a", "choices: choice1, choice2, choice3, choice4"],
        "agent_collaboration": "CoT"
    }
    thinking4b, answer4b = await cot_agent_4b([taskInfo, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4b.id}, quantitatively compare radius ratio with choices, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])
    
    debate_instruction_4c = "Sub-task 4c: Perform a fact-checking and self-consistency check by having multiple independent agents review the mapping from the computed radius ratio to the selected multiple-choice option, and confirm the robustness of the final answer choice."
    debate_agents_4c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4c = self.max_round
    all_thinking4c = [[] for _ in range(N_max_4c)]
    all_answer4c = [[] for _ in range(N_max_4c)]
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": debate_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4c):
        for i, agent in enumerate(debate_agents_4c):
            if r == 0:
                thinking4c, answer4c = await agent([taskInfo, thinking4b, answer4b], debate_instruction_4c, r, is_sub_task=True)
            else:
                input_infos_4c = [taskInfo, thinking4b, answer4b] + all_thinking4c[r-1] + all_answer4c[r-1]
                thinking4c, answer4c = await agent(input_infos_4c, debate_instruction_4c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, fact-check mapping and confirm final answer, thinking: {thinking4c.content}; answer: {answer4c.content}")
            all_thinking4c[r].append(thinking4c)
            all_answer4c[r].append(answer4c)
    final_decision_agent_4c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4c, answer4c = await final_decision_agent_4c([taskInfo] + all_thinking4c[-1] + all_answer4c[-1], "Sub-task 4c: Make final decision on the closest multiple-choice option.", is_sub_task=True)
    agents.append(f"Final Decision agent, confirming final answer choice, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking4c, answer4c, sub_tasks, agents)
    return final_answer, logs
