async def forward_156(self, taskInfo):
    logs = []

    debate_instruction1 = "Sub-task 1: Extract and characterize essential features of the retrovirus and outbreak context, including viral genome type, diagnostic targets (nucleic acid vs antibodies), and constraints such as speed and accuracy."
    final_decision_instruction1 = "Sub-task 1: Provide a comprehensive characterization of the retrovirus and outbreak context."
    debate_desc1 = {
        "instruction": debate_instruction1,
        "final_decision_instruction": final_decision_instruction1,
        "input": [taskInfo],
        "context_desc": ["user query"],
        "temperature": 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, analyze and classify possible identification methods (DNA sequencing, cDNA sequencing, antibody detection, symptom-based inference) and corresponding diagnostic techniques (PCR variants, ELISA), assessing their biological relevance and feasibility for retrovirus detection."
    final_decision_instruction2 = "Sub-task 2: Synthesize and choose the most consistent classification and analysis of identification and diagnostic methods."
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = "Sub-task 3: Transform the classified identification and detection methods into concrete diagnostic kit design variants, including PCR-based kits (standard, nested, real-time) and ELISA kits, considering retrovirus biology and diagnostic performance criteria."
    final_decision_instruction3 = "Sub-task 3: Provide detailed diagnostic kit design variants based on analysis from Sub-task 2."
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Evaluate and prioritize the diagnostic kit design variants based on criteria such as specificity, sensitivity, speed, and suitability for retrovirus detection, to select the optimal molecular diagnostic kit design."
    final_decision_instruction4 = "Sub-task 4: Select the optimal molecular diagnostic kit design based on evaluation criteria."
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
