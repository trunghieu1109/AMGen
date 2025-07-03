async def forward_0(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1a = "Sub-task 1a: Retrieve precise molecular identifiers (SMILES, InChI) and 3D coordinate data or database links for each molecule: quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. Include known symmetry-related structural features."
    cot_agent_1a, thinking1a, answer1a, subtask_desc1a = await self.cot(subtask_id="subtask_1a", cot_instruction=cot_instruction_1a, input_list=[taskInfo], output_fields=["thinking", "answer"], temperature=0.0, context="user input")
    agents.append(f"CoT agent {cot_agent_1a.id}, retrieving molecular identifiers and 3D data, thinking: {thinking1a.content}; answer: {answer1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking1a.content}; answer - {answer1a.content}")
    logs.append(subtask_desc1a)

    cot_instruction_1b = "Sub-task 1b: Using the retrieved molecular data, generate multiple plausible 3D conformations for each molecule to capture structural variability relevant to symmetry analysis."
    cot_agents_1b, thinking1b, answer1b, subtask_desc1b, thinkingmapping1b, answermapping1b = await self.sc_cot(subtask_id="subtask_1b", cot_sc_instruction=cot_instruction_1b, input_list=[taskInfo, thinking1a, answer1a], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1a", "answer of subtask 1a"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking1b.content}; answer - {answer1b.content}")
    for idx, key in enumerate(thinkingmapping1b):
        agents.append(f"CoT-SC agent {cot_agents_1b[idx].id}, generating 3D conformations, thinking: {thinkingmapping1b[key]}; answer: {answermapping1b[key]}")
    logs.append(subtask_desc1b)

    cot_instruction_2 = "Sub-task 2: For each generated 3D conformation, compute the point-group symmetry using established computational methods or software, explicitly listing detected symmetry elements, focusing on identifying C3h symmetry elements (C3 rotation axis plus horizontal mirror plane)."
    cot_agents_2, thinking2, answer2, subtask_desc2, thinkingmapping2, answermapping2 = await self.sc_cot(subtask_id="subtask_2", cot_sc_instruction=cot_instruction_2, input_list=[taskInfo, thinking1b, answer1b], output_fields=["thinking", "answer"], temperature=0.5, context=["user query", "thinking of subtask 1b", "answer of subtask 1b"], n_repeat=self.max_sc)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    for idx, key in enumerate(thinkingmapping2):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, computing point-group symmetries, thinking: {thinkingmapping2[key]}; answer: {answermapping2[key]}")
    logs.append(subtask_desc2)

    debate_instruction_3 = "Sub-task 3: Debate among agents to argue and cross-validate which molecule(s) exhibit C3h symmetry based on computed symmetries, structural data, and known chemical principles. Agents should present supporting evidence and counterarguments for each candidate molecule."
    final_decision_instruction_3 = "Sub-task 3: Make a final decision on which molecule has C3h symmetry, justifying the choice with evidence from the debate."
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
    debate_agents_3, final_decision_agent_3, thinking3, answer3, subtask_desc3, all_thinking3, all_answer3 = await self.debate(subtask_id="subtask_3", debate_desc=debate_desc_3, final_decision_desc=final_decision_desc_3, n_repeat=self.max_round)
    for r in range(self.max_round):
        for idx, agent in enumerate(debate_agents_3):
            agents.append(f"Debate agent {agent.id}, round {r}, arguing C3h symmetry presence, thinking: {all_thinking3[r][idx].content}; answer: {all_answer3[r][idx].content}")
    agents.append(f"Final Decision agent {final_decision_agent_3.id}, concluding C3h symmetry assignment, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)

    cot_reflect_instruction_4 = "Sub-task 4: Review and refine the final answer formatting, ensuring it is a single letter choice (A, B, C, or D) corresponding to the molecule with C3h symmetry, and verify adherence to output format and correctness."
    critic_instruction_4 = "Please critique the final formatted answer for correctness, clarity, and adherence to the specified output format."
    cot_reflect_desc_4 = {
        'instruction': cot_reflect_instruction_4,
        'input': [taskInfo, thinking3, answer3],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc_4 = {
        'instruction': critic_instruction_4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent_4, critic_agent_4, thinking4, answer4, subtask_desc4, feedbacks4, corrects4, thinkings4, answers4 = await self.reflexion(subtask_id="subtask_4", cot_reflect_desc=cot_reflect_desc_4, critic_desc=critic_desc_4, n_repeat=self.max_round)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, reviewing final answer formatting, thinking: {thinkings4[0].content}; answer: {answers4[0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedbacks4[i].content}; answer: {corrects4[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining final answer, thinking: {thinkings4[i + 1].content}; answer: {answers4[i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs