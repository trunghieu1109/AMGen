async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Understand and write down the given operator components P_x, P_y, and P_z as matrices multiplied by \hbar/2, confirming their forms as Pauli matrices."
    cot_agent_1, thinking_1, answer_1, subtask_desc_1 = await self.cot(subtask_id="subtask_1", cot_instruction=cot_instruction_1, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent_1.id}, understanding operator components, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    logs.append(subtask_desc_1)

    cot_sc_instruction_2 = "Sub-task 2: Define the arbitrary direction vector \vecn lying in the x-z plane, parameterized by an angle \theta, and express the operator \vecP \cdot \vecn as a linear combination of P_x and P_z, based on outputs from Sub-task 1."
    cot_agents_2, thinking_2, answer_2, subtask_desc_2, list_thinking_2, list_answer_2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo, thinking_1, answer_1], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    for idx in range(len(list_thinking_2)):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, defining direction vector and operator, thinking: {list_thinking_2[idx]}; answer: {list_answer_2[idx]}")
    logs.append(subtask_desc_2)

    cot_instruction_3 = "Sub-task 3: Formulate the eigenvalue equation (\vecP \cdot \vecn) |\psi\rangle = +\hbar/2 |\psi\rangle explicitly as a 2x2 matrix equation using matrices from Sub-task 1 and vector from Sub-task 2."
    debate_agents_3, final_decision_agent_3, thinking_3, answer_3, subtask_desc_3, list_thinking_3, list_answer_3 = await self.debate(subtask_id="subtask_3", debate_desc={
        "instruction": cot_instruction_3,
        "context": ["user query", thinking_2.content, answer_2.content],
        "input": [taskInfo, thinking_2, answer_2],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }, final_decision_desc={
        "instruction": "Sub-task 3: Make final decision on the explicit eigenvalue matrix equation.",
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_3):
            agents.append(f"Debate agent {agent.id}, round {round}, formulating eigenvalue matrix equation, thinking: {list_thinking_3[round][idx].content}; answer: {list_answer_3[round][idx].content}")
    agents.append(f"Final Decision agent, finalizing eigenvalue matrix equation, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    logs.append(subtask_desc_3)

    cot_instruction_4 = "Sub-task 4: Solve the eigenvalue equation to find the eigenvector corresponding to eigenvalue +\hbar/2, expressing eigenvector components in terms of \theta."
    debate_agents_4, final_decision_agent_4, thinking_4, answer_4, subtask_desc_4, list_thinking_4, list_answer_4 = await self.debate(subtask_id="subtask_4", debate_desc={
        "instruction": cot_instruction_4,
        "context": ["user query", thinking_3.content, answer_3.content],
        "input": [taskInfo, thinking_3, answer_3],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }, final_decision_desc={
        "instruction": "Sub-task 4: Make final decision on the eigenvector solution.",
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_4):
            agents.append(f"Debate agent {agent.id}, round {round}, solving eigenvalue equation, thinking: {list_thinking_4[round][idx].content}; answer: {list_answer_4[round][idx].content}")
    agents.append(f"Final Decision agent, finalizing eigenvector solution, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    logs.append(subtask_desc_4)

    cot_instruction_5 = "Sub-task 5: Normalize the eigenvector obtained in Sub-task 4 to ensure it has unit norm."
    debate_agents_5, final_decision_agent_5, thinking_5, answer_5, subtask_desc_5, list_thinking_5, list_answer_5 = await self.debate(subtask_id="subtask_5", debate_desc={
        "instruction": cot_instruction_5,
        "context": ["user query", thinking_4.content, answer_4.content],
        "input": [taskInfo, thinking_4, answer_4],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }, final_decision_desc={
        "instruction": "Sub-task 5: Make final decision on the normalized eigenvector.",
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_5):
            agents.append(f"Debate agent {agent.id}, round {round}, normalizing eigenvector, thinking: {list_thinking_5[round][idx].content}; answer: {list_answer_5[round][idx].content}")
    agents.append(f"Final Decision agent, finalizing normalized eigenvector, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    logs.append(subtask_desc_5)

    cot_instruction_6 = "Sub-task 6: Compare the normalized eigenvector components with the given multiple-choice options and select the correct choice that matches the normalized eigenvector."
    debate_agents_6, final_decision_agent_6, thinking_6, answer_6, subtask_desc_6, list_thinking_6, list_answer_6 = await self.debate(subtask_id="subtask_6", debate_desc={
        "instruction": cot_instruction_6,
        "context": ["user query", thinking_5.content, answer_5.content],
        "input": [taskInfo, thinking_5, answer_5],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }, final_decision_desc={
        "instruction": "Sub-task 6: Make final decision on the correct multiple-choice eigenvector.",
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_6):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing eigenvector with choices, thinking: {list_thinking_6[round][idx].content}; answer: {list_answer_6[round][idx].content}")
    agents.append(f"Final Decision agent, finalizing correct choice selection, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    logs.append(subtask_desc_6)

    final_answer = await self.make_final_answer(thinking_6, answer_6, sub_tasks, agents)
    return final_answer, logs
