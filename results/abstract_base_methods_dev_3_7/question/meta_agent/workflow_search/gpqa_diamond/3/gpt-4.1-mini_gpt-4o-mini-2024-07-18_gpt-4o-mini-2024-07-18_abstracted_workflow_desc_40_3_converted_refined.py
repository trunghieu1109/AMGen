async def forward_3(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: List the standard Maxwell's equations in differential form, focusing on divergence and curl of E and B fields. Cross-verify multiple consistent listings to ensure completeness and correctness."
    cot_agents_1, thinking1_sc, answer1_sc, subtask_desc1_sc, list_thinking1, list_answer1 = await self.sc_cot(subtask_id="subtask_1", cot_sc_instruction=cot_sc_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {cot_agents_1[idx].id}, listing standard Maxwell's equations variant {idx+1}, thinking: {list_thinking1[idx]}; answer: {list_answer1[idx]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_sc.content}; answer - {answer1_sc.content}")
    logs.append(subtask_desc1_sc)

    cot_reflect_instruction_2 = "Sub-task 2: Analyze how the existence of magnetic monopoles modifies Maxwell's equations. After your reasoning, write out both the standard Maxwell's equations and the symmetric Maxwell's equations including magnetic charge and current densities. Explicitly circle which two equations differ due to magnetic monopoles, clarifying that only divergence of B and curl of E change, not curl of B."
    critic_instruction_2 = "Please review the analysis and highlight any misinterpretations, especially regarding which curl equation changes with magnetic monopoles."
    cot_reflect_desc_2 = {
        'instruction': cot_reflect_instruction_2,
        'input': [taskInfo, thinking1_sc, answer1_sc],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc_2 = {
        'instruction': critic_instruction_2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent2, critic_agent2, thinking2, answer2, subtask_desc2, list_feedback2, list_correct2, list_thinking2, list_answer2 = await self.reflexion(subtask_id="subtask_2", cot_reflect_desc=cot_reflect_desc_2, critic_desc=critic_desc_2, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, analyzing modifications due to magnetic monopoles, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent2.id}, providing feedback, thinking: {list_feedback2[i].content}; answer: {list_correct2[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refining analysis, thinking: {list_thinking2[i + 1].content}; answer: {list_answer2[i + 1].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(subtask_desc2)

    cot_sc_instruction_3 = "Sub-task 3: List two independent derivations of Maxwell's equations with magnetic monopoles: one from duality symmetry and one from direct extension. For each, identify which equations change. Vote on the consistent modifications, focusing on divergence of B and curl of E."
    cot_agents_3, thinking3_sc, answer3_sc, subtask_desc3_sc, list_thinking3, list_answer3 = await self.sc_cot(subtask_id="subtask_3", cot_sc_instruction=cot_sc_instruction_3, input_list=[taskInfo, thinking2, answer2], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", thinking2.content, answer2.content], n_repeat=self.max_sc)
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {cot_agents_3[idx].id}, derivation variant {idx+1}, thinking: {list_thinking3[idx]}; answer: {list_answer3[idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_sc.content}; answer - {answer3_sc.content}")
    logs.append(subtask_desc3_sc)

    debate_instruction_4 = "Sub-task 4: Debate which curl equation changes due to magnetic monopoles. One agent defends that curl of E changes (∇×E = -∂B/∂t - J_m), quoting the full modified Faraday's law. Another agent defends that curl of B changes. Each must provide explicit differential forms and argue their correctness."
    final_decision_instruction_4 = "Sub-task 4: Make final decision on which curl equation changes due to magnetic monopoles, based on the debate."
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", thinking3_sc.content, answer3_sc.content],
        "input": [taskInfo, thinking3_sc, answer3_sc],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_4 = {
        "instruction": final_decision_instruction_4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents_4, final_decision_agent_4, thinking4, answer4, subtask_desc4, list_thinking4, list_answer4 = await self.debate(subtask_id="subtask_4", debate_desc=debate_desc_4, final_decision_desc=final_decision_desc_4, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_4):
            agents.append(f"Debate agent {agent.id}, round {round}, debating curl equation change, thinking: {list_thinking4[round][idx].content}; answer: {list_answer4[round][idx].content}")
    agents.append(f"Final Decision agent, deciding curl equation change, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    cot_instruction_5 = "Sub-task 5: Map the identified modified Maxwell's equations (divergence of B and curl of E) to the provided multiple-choice options. Explicitly annotate each choice with the Maxwell operator it refers to, then select the correct choice that matches the two changed equations."
    cot_agent_5, thinking5, answer5, subtask_desc5 = await self.cot(subtask_id="subtask_5", cot_instruction=cot_instruction_5, input_list=[taskInfo, thinking4, answer4], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", thinking4.content, answer4.content])
    agents.append(f"CoT agent {cot_agent_5.id}, mapping modified equations to choices, thinking: {thinking5.content}; answer: {answer5.content}")

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs