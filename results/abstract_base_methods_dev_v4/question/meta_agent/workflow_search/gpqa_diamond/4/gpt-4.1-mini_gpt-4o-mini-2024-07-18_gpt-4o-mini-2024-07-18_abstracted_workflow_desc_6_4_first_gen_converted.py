async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Define the arbitrary direction vector \vecn lying in the x-z plane in terms of an angle \theta, such that \vecn = (sin\theta, 0, cos\theta). This sets the basis for constructing the operator along \vecn." 
    cot_agent_1, thinking1, answer1, subtask_desc1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent_1.id}, defining direction vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = "Sub-task 2: Construct the operator P_n = \vecP \cdot \vecn = P_x sin\theta + P_y * 0 + P_z cos\theta using the given matrices for P_x, P_y, and P_z. This operator acts on the spinor space and is a 2x2 matrix." 
    cot_agents_2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo, thinking1, answer1], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx, key in enumerate(list_thinking2):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, constructing operator P_n, thinking: {list_thinking2[key]}; answer: {list_answer2[key]}")
    logs.append(subtask_desc2)
    cot_reflect_instruction_3 = "Sub-task 3: Set up the eigenvalue equation P_n |v> = (+\hbar/2) |v> using the operator from subtask_2 and solve for the eigenvector |v> in terms of components (v_1, v_2)."
    critic_instruction_3 = "Please review the eigenvector solution and provide its limitations."
    cot_reflect_desc_3 = {
        'instruction': cot_reflect_instruction_3, 'input': [taskInfo, thinking1, answer1, thinking2, answer2], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc_3 = {
        'instruction': critic_instruction_3, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    cot_agent_3, critic_agent_3, thinking3, answer3, subtask_desc3, list_feedback3, list_correct3, list_thinking3, list_answer3 = await self.reflexion(subtask_id="subtask_3", cot_reflect_desc=cot_reflect_desc_3, critic_desc=critic_desc_3, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, solving eigenvalue equation, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {list_feedback3[i].content}; answer: {list_correct3[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining eigenvector solution, thinking: {list_thinking3[i + 1].content}; answer: {list_answer3[i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)
    cot_reflect_instruction_4 = "Sub-task 4: Normalize the eigenvector |v> obtained in subtask_3 to ensure it has unit norm, yielding the normalized eigenvector elements."
    critic_instruction_4 = "Please review the normalization process and provide its limitations."
    cot_reflect_desc_4 = {
        'instruction': cot_reflect_instruction_4, 'input': [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], 'output': ["thinking", "answer"],
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc_4 = {
        'instruction': critic_instruction_4, 'output': ["feedback", "correct"], 'temperature': 0.0
    }
    cot_agent_4, critic_agent_4, thinking4, answer4, subtask_desc4, list_feedback4, list_correct4, list_thinking4, list_answer4 = await self.reflexion(subtask_id="subtask_4", cot_reflect_desc=cot_reflect_desc_4, critic_desc=critic_desc_4, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, normalizing eigenvector, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {list_feedback4[i].content}; answer: {list_correct4[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining normalization, thinking: {list_thinking4[i + 1].content}; answer: {list_answer4[i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)
    debate_instruction_5 = "Sub-task 5: Compare the normalized eigenvector elements with the given multiple-choice options to identify the correct form of the eigenvector corresponding to eigenvalue +\hbar/2 along \vecn in the x-z plane."
    final_decision_instruction_5 = "Sub-task 5: Make final decision on the correct normalized eigenvector form from the given options."
    debate_desc_5 = {
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "input": [taskInfo, thinking4, answer4],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_5 = {
        "instruction": final_decision_instruction_5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents_5, final_decision_agent_5, thinking5, answer5, subtask_desc5, list_thinking5, list_answer5 = await self.debate(subtask_id="subtask_5", debate_desc=debate_desc_5, final_decision_desc=final_decision_desc_5, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_5):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing eigenvector with options, thinking: {list_thinking5[round][idx].content}; answer: {list_answer5[round][idx].content}")
    agents.append(f"Final Decision agent, selecting correct eigenvector, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    logs.append(subtask_desc5)
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs