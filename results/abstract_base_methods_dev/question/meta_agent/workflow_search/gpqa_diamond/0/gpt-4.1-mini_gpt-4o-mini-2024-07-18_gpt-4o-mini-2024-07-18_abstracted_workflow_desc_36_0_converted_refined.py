async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction_1 = (
        "Sub-task 1: For each molecule (triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, "
        "and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone), produce three independent analyses that: "
        "(a) specify all potential symmetry axes, (b) test for horizontal and vertical mirror planes, and (c) conclude its point group. "
        "Then vote to select the most consistent point group result for each molecule, with detailed structural reasoning. "
        "Use the context from the task information for guidance."
    )
    cot_agents1, thinking1, answer1, subtask_desc1, list_thinking1, list_answer1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_sc_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    for idx, key in enumerate(list_thinking1):
        agents.append(f"CoT-SC agent {cot_agents1[idx].id}, independent analysis {idx+1}, thinking: {list_thinking1[key]}; answer: {list_answer1[key]}")
    logs.append(subtask_desc1)
    debate_instruction_2 = (
        "Sub-task 2: Based on the multiple analyses from Sub-task 1, Agent A asserts each molecule's point group assignment. "
        "Agents B and C critique or defend these assignments by applying specific symmetry checks, including verifying symmetry axes, mirror planes, and other elements. "
        "Agents then debate to converge on an agreed point group for each molecule, ensuring systematic point-group determination."
    )
    final_decision_instruction_2 = (
        "Sub-task 2: Make a final consensus decision on the point group of each molecule after debate."
    )
    debate_desc_2 = {
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "input": [taskInfo, thinking1, answer1],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_2 = {
        "instruction": final_decision_instruction_2,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents2, final_decision_agent2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc_2,
        final_decision_desc=final_decision_desc_2,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents2):
            agents.append(f"Debate agent {agent.id}, round {round}, debating point group assignments, thinking: {list_thinking2[round][idx].content}; answer: {list_answer2[round][idx].content}")
    agents.append(f"Final Decision agent, consensus on point groups, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(subtask_desc2)
    cot_reflect_instruction_3 = (
        "Sub-task 3: Based on the consensus point groups from Sub-task 2, identify which molecule exhibits C3h symmetry. "
        "Map this molecule to the corresponding multiple-choice answer (A, B, C, or D). "
        "Then reflect on whether the assigned point group truly matches C3h criteria and if the molecule matches any documented textbook example of C3h symmetry. "
        "Provide a detailed justification and highlight any limitations or uncertainties."
    )
    critic_instruction_3 = (
        "Please review the mapping of the molecule to C3h symmetry and the multiple-choice answer. "
        "Provide feedback on the correctness and any potential errors or oversights."
    )
    cot_reflect_desc_3 = {
        'instruction': cot_reflect_instruction_3,
        'input': [taskInfo, thinking1, answer1, thinking2, answer2],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc_3 = {
        'instruction': critic_instruction_3,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    cot_agent3, critic_agent3, thinking3, answer3, subtask_desc3, list_feedback3, list_correct3, list_thinking3, list_answer3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc_3,
        critic_desc=critic_desc_3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, reflecting on final molecule assignment, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent3.id}, feedback round {i}, thinking: {list_feedback3[i].content}; answer: {list_correct3[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining reflection round {i}, thinking: {list_thinking3[i + 1].content}; answer: {list_answer3[i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs