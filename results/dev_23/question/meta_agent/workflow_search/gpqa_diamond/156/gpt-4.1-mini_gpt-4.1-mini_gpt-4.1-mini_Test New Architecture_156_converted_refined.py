async def forward_156(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Confirm and characterize the viral genome type and molecular targets relevant for diagnostic design, "
        "explicitly verifying that the retrovirus has an RNA genome requiring reverse transcription. "
        "Critically evaluate the presence of viral RNA, proviral DNA, and antibody markers, and clarify the implications for sequencing and detection methods. "
        "Explicitly reject DNA sequencing as a primary identification method if no DNA virus is present, addressing the previous failure of conflating DNA sequencing with cDNA/RT-PCR."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the confirmed viral genome and molecular targets from subtask_1, "
        "analyze and compare the suitability of different sequencing approaches (DNA sequencing, cDNA sequencing, direct RNA detection) "
        "and diagnostic methods (PCR variants, ELISA, symptom-based inference). Include turnaround time, technical feasibility, and biological appropriateness, "
        "explicitly contrasting DNA sequencing versus cDNA sequencing for an RNA virus and considering the need for reverse transcription steps."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, critically evaluate the impact of each diagnostic approach on speed, accuracy, sensitivity, specificity, and feasibility, "
        "including the reliability of antibody detection versus direct viral detection. Incorporate a critical review of assumptions made in previous subtasks, "
        "explicitly challenging any default acceptance of DNA sequencing or PCR without reverse transcription, to prevent groupthink and premature convergence on incorrect options."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Integrate insights from the viral genome characterization, sequencing and diagnostic method analysis, and critical evaluation to select the optimal molecular diagnostic approach and design strategy for the kit. "
        "Justify the choice based on biological appropriateness (e.g., RNA genome requiring cDNA sequencing and real-time RT-PCR), practical considerations (speed, accuracy), "
        "and explicitly reject unsuitable options such as DNA sequencing-first strategies. The design must prioritize quick and accurate detection aligned with retroviral biology."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
