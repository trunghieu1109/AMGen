async def forward_0(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Retrieve the SMILES strings and available 3D coordinates for each molecule: quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone from reliable chemical databases (e.g., PubChem, ChemSpider). Provide data in standard formats (SMILES, MOL, or PDB)."
    cot_agents_1, thinking_1, answer_1, subtask_desc_1, thinkingmapping_1, answermapping_1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(thinkingmapping_1):
        agents.append(f"CoT-SC agent {cot_agents_1[idx].id}, retrieving molecular data, thinking: {thinkingmapping_1[key]}; answer: {answermapping_1[key]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    logs.append(subtask_desc_1)

    cot_instruction_2 = "Sub-task 2: Generate or retrieve optimized 3D geometries for each molecule using computational chemistry tools (e.g., RDKit conformer generation, DFT calculations). Consider multiple conformers to ensure accurate geometry for symmetry analysis."
    cot_agents_2, thinking_2, answer_2, subtask_desc_2, thinkingmapping_2, answermapping_2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_instruction_2,
        input_list=[thinking_1, answer_1],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["molecular data from subtask 1"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(thinkingmapping_2):
        agents.append(f"CoT-SC agent {cot_agents_2[idx].id}, generating optimized geometries, thinking: {thinkingmapping_2[key]}; answer: {answermapping_2[key]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    logs.append(subtask_desc_2)

    cot_instruction_3 = "Sub-task 3: Identify and analyze symmetry elements (e.g., rotation axes, mirror planes) of each molecule's optimized geometry using automated tools (e.g., OpenBabel symmetry finder). Validate symmetry assignments with literature references where available."
    critic_instruction_3 = "Please review the symmetry element identification for accuracy and highlight any inconsistencies or missing validations."
    cot_reflect_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo, thinking_2, answer_2],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "optimized geometries"]
    }
    critic_desc_3 = {
        'instruction': critic_instruction_3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    cot_agent_3, critic_agent_3, thinking_3, answer_3, subtask_desc_3, feedbacks_3, corrects_3, thinkings_3, answers_3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc_3,
        critic_desc=critic_desc_3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, identifying symmetry elements, thinking: {thinkings_3[0].content}; answer: {answers_3[0].content}")
    for i in range(self.max_round):
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedbacks_3[i].content}; answer: {corrects_3[i].content}")
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining symmetry identification, thinking: {thinkings_3[i + 1].content}; answer: {answers_3[i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    logs.append(subtask_desc_3)

    cot_instruction_4 = "Sub-task 4: Cross-validate the identified symmetry results with authoritative literature or quantum-chemical calculations to confirm or refute C3h symmetry assignment for each molecule."
    debate_instruction_4 = "Sub-task 4: Debate any conflicting symmetry validation results among agents and reach consensus on the correct symmetry classification."
    final_decision_instruction_4 = "Sub-task 4: Make final decision on the confirmed symmetry classification for each molecule based on cross-validation and debate outcomes."
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", thinking_3, answer_3],
        "input": [taskInfo, thinking_3, answer_3],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_4 = {
        "instruction": final_decision_instruction_4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    debate_agents_4, final_decision_agent_4, thinking_4, answer_4, subtask_desc_4, all_thinking_4, all_answer_4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc=final_decision_desc_4,
        n_repeat=self.max_round
    )
    for r in range(self.max_round):
        for idx, agent in enumerate(debate_agents_4):
            agents.append(f"Debate agent {agent.id}, round {r}, debating symmetry validation, thinking: {all_thinking_4[r][idx].content}; answer: {all_answer_4[r][idx].content}")
    agents.append(f"Final Decision agent, confirming symmetry classification, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    logs.append(subtask_desc_4)

    cot_instruction_5 = "Sub-task 5: Based on the confirmed symmetry classifications, select the correct multiple-choice letter (A-D) corresponding to the molecule that has C3h symmetry. Ensure the answer is exactly one letter and justified by the validated data."
    cot_agent_5, thinking_5, answer_5, subtask_desc_5 = await self.cot(
        subtask_id="subtask_5",
        cot_instruction=cot_instruction_5,
        input_list=[taskInfo, thinking_4, answer_4],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "confirmed symmetry classification"]
    )
    agents.append(f"CoT agent {cot_agent_5.id}, selecting final answer, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    logs.append(subtask_desc_5)

    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs