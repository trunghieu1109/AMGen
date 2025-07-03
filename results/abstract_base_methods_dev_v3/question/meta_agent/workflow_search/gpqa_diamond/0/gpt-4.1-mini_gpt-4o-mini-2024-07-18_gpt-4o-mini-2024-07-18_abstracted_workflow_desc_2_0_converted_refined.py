async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_reflect_instruction_1 = "Sub-task 1: Retrieve and verify the 3D geometries and documented symmetry point groups for each molecule: triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. Cross-check the retrieved symmetry information against authoritative chemical databases or literature to ensure accuracy."
    critic_instruction_1 = "Please review the retrieved molecular symmetry data and verify its correctness, pointing out any inconsistencies or missing verification."
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1,
        'input': [taskInfo],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query']
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    cot_agent_1, critic_agent_1, thinking1, answer1, subtask_desc1, list_feedback1, list_correct1, list_thinking1, list_answer1 = await self.reflexion(subtask_id="subtask_1", cot_reflect_desc=cot_reflect_desc_1, critic_desc=critic_desc_1, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, retrieving and verifying molecular symmetry data, thinking: {thinking1.content}; answer: {answer1.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent_1.id}, providing feedback, thinking: {list_feedback1[i].content}; answer: {list_correct1[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining molecular symmetry data, thinking: {list_thinking1[i + 1].content}; answer: {list_answer1[i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = "Sub-task 2: Analyze the molecular symmetry elements of each molecule based on the verified structures and symmetry data from Sub-task 1. Explore multiple reasoning paths to identify the presence or absence of C3 and horizontal mirror plane (h) symmetry elements, and determine the point groups with self-consistency and cross-verification from multiple sources."
    cot_agents_2, thinking2, answer2, subtask_desc2, list_thinking2, list_answer2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_sc_instruction_2, input_list=[taskInfo, thinking1, answer1], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1", "answer of subtask 1"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx, key in enumerate(list_thinking2):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, analyzing symmetry elements with multiple reasoning paths, thinking: {list_thinking2[key]}; answer: {list_answer2[key]}")
    logs.append(subtask_desc2)
    debate_instruction_3 = "Sub-task 3: Debate the validity of the identified point groups for all four molecules from Sub-task 2, focusing on whether each molecule matches the C3h point group. Challenge assumptions and reasoning paths to reach a consensus on the correct molecule(s) with C3h symmetry."
    final_decision_instruction_3 = "Sub-task 3: Make final decision on which molecule(s) have C3h symmetry based on the debate outcomes."
    debate_desc_3 = {
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "input": [taskInfo, thinking2, answer2],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_3 = {
        "instruction": final_decision_instruction_3,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents_3, final_decision_agent_3, thinking3, answer3, subtask_desc3, list_thinking3, list_answer3 = await self.debate(subtask_id="subtask_3", debate_desc=debate_desc_3, final_decision_desc=final_decision_desc_3, n_repeat=self.max_round)
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_agents_3):
            agents.append(f"Debate agent {agent.id}, round {round}, debating point group validity, thinking: {list_thinking3[round][idx].content}; answer: {list_answer3[round][idx].content}")
    agents.append(f"Final Decision agent, concluding molecule(s) with C3h symmetry, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)
    cot_reflect_instruction_4 = "Sub-task 4: Reflect on the final identified molecule(s) with C3h symmetry from Sub-task 3 and validate the mapping to the corresponding multiple-choice letter (A, B, C, or D). Ensure the final answer aligns with the verified symmetry analysis and the query requirements."
    critic_instruction_4 = "Please review the final mapping of the molecule(s) to the multiple-choice letter and provide feedback on its correctness and consistency with the symmetry analysis."
    cot_reflect_desc_4 = {
        'instruction': cot_reflect_instruction_4,
        'input': [taskInfo, thinking3, answer3],
        'output': ['thinking', 'answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 3', 'answer of subtask 3']
    }
    critic_desc_4 = {
        'instruction': critic_instruction_4,
        'output': ['feedback', 'correct'],
        'temperature': 0.0
    }
    cot_agent_4, critic_agent_4, thinking4, answer4, subtask_desc4, list_feedback4, list_correct4, list_thinking4, list_answer4 = await self.reflexion(subtask_id="subtask_4", cot_reflect_desc=cot_reflect_desc_4, critic_desc=critic_desc_4, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, validating final mapping to choice letter, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {list_feedback4[i].content}; answer: {list_correct4[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining final mapping, thinking: {list_thinking4[i + 1].content}; answer: {list_answer4[i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs