async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Analyze the given spin state |ψ⟩ = 0.5|↑⟩ + (√3)/2|↓⟩ and identify the basis states |↑⟩ and |↓⟩ as eigenstates of σ_z. Express the state in vector form suitable for calculations."
    cot_agents_1, thinking1, answer1, subtask_desc1, list_thinking1, list_answer1 = await self.sc_cot(subtask_id="subtask_1", cot_sc_instruction=cot_sc_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.5, context=["user query"], n_repeat=self.max_sc)
    cot_reflect_desc_1 = {
        'instruction': "Sub-task 1 Reflexion: Review the vector form and basis identification for correctness and clarity.",
        'input': [taskInfo, thinking1, answer1],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc_1 = {
        'instruction': "Please review the vector form and basis identification and provide feedback and corrections if needed.",
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent1, critic_agent1, thinking1_ref, answer1_ref, subtask_desc1_ref, list_feedback1, list_correct1, list_thinking1_ref, list_answer1_ref = await self.reflexion(subtask_id="subtask_1_reflexion", cot_reflect_desc=cot_reflect_desc_1, critic_desc=critic_desc_1, n_repeat=self.max_round)
    agents.append(f"SC_CoT agent {cot_agents_1[0].id}, analyzing spin state, thinking: {thinking1.content}; answer: {answer1.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent1.id}, feedback round {i}, thinking: {list_feedback1[i].content}; answer: {list_correct1[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent1.id}, refining vector form round {i}, thinking: {list_thinking1_ref[i+1].content}; answer: {list_answer1_ref[i+1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_ref.content}; answer - {answer1_ref.content}")
    logs.append(subtask_desc1_ref)

    cot_sc_instruction_2z = "Sub-task 2: Calculate the expectation value ⟨ψ|σ_z|ψ⟩ using the vector form of |ψ⟩ and the matrix representation of σ_z."
    cot_agents_2z, thinking2z, answer2z, subtask_desc2z, list_thinking2z, list_answer2z = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2z, input_list=[taskInfo, thinking1_ref, answer1_ref], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    cot_reflect_desc_2z = {
        'instruction': "Sub-task 2 Reflexion: Review the calculation of expectation value ⟨ψ|σ_z|ψ⟩ for accuracy.",
        'input': [taskInfo, thinking1_ref, answer1_ref, thinking2z, answer2z],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc_2z = {
        'instruction': "Please review the expectation value ⟨ψ|σ_z|ψ⟩ calculation and provide feedback and corrections if needed.",
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent2z, critic_agent2z, thinking2z_ref, answer2z_ref, subtask_desc2z_ref, list_feedback2z, list_correct2z, list_thinking2z_ref, list_answer2z_ref = await self.reflexion(subtask_id="subtask_2_reflexion", cot_reflect_desc=cot_reflect_desc_2z, critic_desc=critic_desc_2z, n_repeat=self.max_round)
    agents.append(f"SC_CoT agent {cot_agents_2z[0].id}, calculating ⟨ψ|σ_z|ψ⟩, thinking: {thinking2z.content}; answer: {answer2z.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent2z.id}, feedback round {i}, thinking: {list_feedback2z[i].content}; answer: {list_correct2z[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent2z.id}, refining ⟨ψ|σ_z|ψ⟩ round {i}, thinking: {list_thinking2z_ref[i+1].content}; answer: {list_answer2z_ref[i+1].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2z_ref.content}; answer - {answer2z_ref.content}")
    logs.append(subtask_desc2z_ref)

    cot_sc_instruction_2x = "Sub-task 3: Calculate the expectation value ⟨ψ|σ_x|ψ⟩ using the vector form of |ψ⟩ and the matrix representation of σ_x."
    cot_agents_2x, thinking2x, answer2x, subtask_desc2x, list_thinking2x, list_answer2x = await self.sc_cot(subtask_id="subtask_3", cot_sc_instruction=cot_sc_instruction_2x, input_list=[taskInfo, thinking1_ref, answer1_ref], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    cot_reflect_desc_2x = {
        'instruction': "Sub-task 3 Reflexion: Review the calculation of expectation value ⟨ψ|σ_x|ψ⟩ for accuracy.",
        'input': [taskInfo, thinking1_ref, answer1_ref, thinking2x, answer2x],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc_2x = {
        'instruction': "Please review the expectation value ⟨ψ|σ_x|ψ⟩ calculation and provide feedback and corrections if needed.",
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent2x, critic_agent2x, thinking2x_ref, answer2x_ref, subtask_desc2x_ref, list_feedback2x, list_correct2x, list_thinking2x_ref, list_answer2x_ref = await self.reflexion(subtask_id="subtask_3_reflexion", cot_reflect_desc=cot_reflect_desc_2x, critic_desc=critic_desc_2x, n_repeat=self.max_round)
    agents.append(f"SC_CoT agent {cot_agents_2x[0].id}, calculating ⟨ψ|σ_x|ψ⟩, thinking: {thinking2x.content}; answer: {answer2x.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent2x.id}, feedback round {i}, thinking: {list_feedback2x[i].content}; answer: {list_correct2x[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent2x.id}, refining ⟨ψ|σ_x|ψ⟩ round {i}, thinking: {list_thinking2x_ref[i+1].content}; answer: {list_answer2x_ref[i+1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2x_ref.content}; answer - {answer2x_ref.content}")
    logs.append(subtask_desc2x_ref)

    cot_instruction_4 = "Sub-task 4: Combine the expectation values from subtask_2 and subtask_3 to compute the expectation value of the operator 10σ_z + 5σ_x, rounding the result to one decimal place."
    cot_agents_4, thinking4, answer4, subtask_desc4, list_thinking4, list_answer4 = await self.sc_cot(subtask_id="subtask_4", cot_sc_instruction=cot_instruction_4, input_list=[taskInfo, thinking2z_ref, answer2z_ref, thinking2x_ref, answer2x_ref], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"], n_repeat=self.max_sc)
    agents.append(f"CoT-SC agent {cot_agents_4[0].id}, combining expectation values to compute ⟨10σ_z + 5σ_x⟩, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    debate_instruction_5 = "Sub-task 5: Based on the output of Sub-task 4, compare the computed expectation value with the given multiple-choice options and select the correct choice (A, B, C, or D)."
    final_decision_instruction_5 = "Sub-task 5: Make final decision on the correct multiple-choice option based on the computed expectation value."
    debate_desc_5 = {
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content, answer4.content],
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
            agents.append(f"Debate agent {agent.id}, round {round}, comparing computed value with choices, thinking: {list_thinking5[round][idx].content}; answer: {list_answer5[round][idx].content}")
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    logs.append(subtask_desc5)

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
