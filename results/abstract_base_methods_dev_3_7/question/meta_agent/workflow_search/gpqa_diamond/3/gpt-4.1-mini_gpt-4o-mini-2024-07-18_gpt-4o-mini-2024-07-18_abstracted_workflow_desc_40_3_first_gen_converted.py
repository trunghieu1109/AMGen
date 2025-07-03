async def forward_3(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Extract and list the standard Maxwell's equations in their differential form, focusing on the equations involving the magnetic field (B) and electric field (E)."
    cot_agent_1, thinking1, answer1, subtask_desc1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent_1.id}, extracting Maxwell's equations, thinking: {thinking1.content}; answer: {answer1.content}")

    cot_sc_instruction_1 = "Sub-task 1: Based on the initial extraction, consider multiple consistent listings of Maxwell's equations involving B and E to ensure completeness and correctness."
    cot_agents_1, thinking1_sc, answer1_sc, subtask_desc1_sc, list_thinking1, list_answer1 = await self.sc_cot(subtask_id="subtask_1_sc", cot_sc_instruction=cot_sc_instruction_1, input_list=[taskInfo, thinking1, answer1], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {cot_agents_1[idx].id}, listing Maxwell's equations variant {idx+1}, thinking: {list_thinking1[idx]}; answer: {list_answer1[idx]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_sc.content}; answer - {answer1_sc.content}")
    logs.append(subtask_desc1_sc)

    cot_instruction_2a = "Sub-task 2: Analyze the physical meaning of each Maxwell's equation, especially those related to the divergence and curl of the magnetic field and the circulation of the electric field."
    cot_agent_2a, thinking2a, answer2a, subtask_desc2a = await self.cot(subtask_id="subtask_2", cot_instruction=cot_instruction_2a, input_list=[taskInfo, thinking1_sc, answer1_sc], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", "thinking of subtask 1", "answer of subtask 1"])
    agents.append(f"CoT agent {cot_agent_2a.id}, analyzing physical meaning of Maxwell's equations, thinking: {thinking2a.content}; answer: {answer2a.content}")

    cot_sc_instruction_2a = "Sub-task 2: Based on the analysis, consider multiple interpretations of the physical meaning of Maxwell's equations related to magnetic and electric fields."
    cot_agents_2a, thinking2a_sc, answer2a_sc, subtask_desc2a_sc, list_thinking2a, list_answer2a = await self.sc_cot(subtask_id="subtask_2_sc", cot_sc_instruction=cot_sc_instruction_2a, input_list=[taskInfo, thinking1_sc, answer1_sc, thinking2a, answer2a], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], n_repeat=self.max_sc)
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {cot_agents_2a[idx].id}, interpreting physical meaning variant {idx+1}, thinking: {list_thinking2a[idx]}; answer: {list_answer2a[idx]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2a_sc.content}; answer - {answer2a_sc.content}")
    logs.append(subtask_desc2a_sc)

    cot_instruction_2b = "Sub-task 3: Identify how the existence of magnetic monopoles modifies Maxwell's equations, particularly which equations change and how."
    cot_agent_2b, thinking2b, answer2b, subtask_desc2b = await self.cot(subtask_id="subtask_3", cot_instruction=cot_instruction_2b, input_list=[taskInfo, thinking1_sc, answer1_sc], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", "thinking of subtask 1", "answer of subtask 1"])
    agents.append(f"CoT agent {cot_agent_2b.id}, identifying modifications due to magnetic monopoles, thinking: {thinking2b.content}; answer: {answer2b.content}")

    cot_sc_instruction_2b = "Sub-task 3: Based on the identification, consider multiple consistent modifications of Maxwell's equations in presence of magnetic monopoles."
    cot_agents_2b, thinking2b_sc, answer2b_sc, subtask_desc2b_sc, list_thinking2b, list_answer2b = await self.sc_cot(subtask_id="subtask_3_sc", cot_sc_instruction=cot_sc_instruction_2b, input_list=[taskInfo, thinking1_sc, answer1_sc, thinking2b, answer2b], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"], n_repeat=self.max_sc)
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {cot_agents_2b[idx].id}, modifications variant {idx+1}, thinking: {list_thinking2b[idx]}; answer: {list_answer2b[idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2b_sc.content}; answer - {answer2b_sc.content}")
    logs.append(subtask_desc2b_sc)

    debate_instruction_4 = "Sub-task 4: Determine specifically which Maxwell's equations differ in the presence of magnetic monopoles by comparing the standard and modified forms."
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", thinking2a_sc.content, answer2a_sc.content, thinking2b_sc.content, answer2b_sc.content],
        "input": [taskInfo, thinking2a_sc, answer2a_sc, thinking2b_sc, answer2b_sc],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    debate_agents_4, final_decision_agent_4, thinking4, answer4, subtask_desc4, list_thinking4, list_answer4 = await self.debate(subtask_id="subtask_4", debate_desc=debate_desc_4, final_decision_desc={"instruction": "Sub-task 4: Make final decision on which Maxwell's equations differ due to magnetic monopoles.", "output": ["thinking", "answer"], "temperature": 0.0}, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_4):
            agents.append(f"Debate agent {agent.id}, round {round}, determining differing Maxwell's equations, thinking: {list_thinking4[round][idx].content}; answer: {list_answer4[round][idx].content}")
    agents.append(f"Final Decision agent, deciding differing Maxwell's equations, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    cot_instruction_5 = "Sub-task 5: Match the identified differing Maxwell's equations to the provided multiple-choice options and select the correct choice."
    cot_agent_5, thinking5, answer5, subtask_desc5 = await self.cot(subtask_id="subtask_5", cot_instruction=cot_instruction_5, input_list=[taskInfo, thinking4, answer4], output_fields=["thinking", "answer"], temperature=0.0, context=["user query", thinking4.content, answer4.content])
    agents.append(f"CoT agent {cot_agent_5.id}, matching differing equations to choices, thinking: {thinking5.content}; answer: {answer5.content}")

    cot_sc_instruction_5 = "Sub-task 5: Based on the matching, consider multiple consistent selections of the correct choice for differing Maxwell's equations."
    cot_agents_5, thinking5_sc, answer5_sc, subtask_desc5_sc, list_thinking5, list_answer5 = await self.sc_cot(subtask_id="subtask_5_sc", cot_sc_instruction=cot_sc_instruction_5, input_list=[taskInfo, thinking4, answer4, thinking5, answer5], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", thinking4.content, answer4.content, thinking5.content, answer5.content], n_repeat=self.max_sc)
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {cot_agents_5[idx].id}, choice selection variant {idx+1}, thinking: {list_thinking5[idx]}; answer: {list_answer5[idx]}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5_sc.content}; answer - {answer5_sc.content}")
    logs.append(subtask_desc5_sc)

    final_answer = await self.make_final_answer(thinking5_sc, answer5_sc, sub_tasks, agents)
    return final_answer, logs