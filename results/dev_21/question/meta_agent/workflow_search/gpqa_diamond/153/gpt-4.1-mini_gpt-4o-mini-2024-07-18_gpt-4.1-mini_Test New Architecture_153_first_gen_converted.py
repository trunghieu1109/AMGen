async def forward_153(self, taskInfo):
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze the mass spectrometry data to identify molecular weight and isotope patterns indicative of elements present (e.g., chlorine), with context from taskInfo."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = "Sub-task 2: Analyze the IR spectrum to identify key functional groups present in the compound, focusing on O-H and C=O stretches, with context from taskInfo."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = "Sub-task 3: Analyze the 1H NMR data to determine the proton environment, including aromatic substitution pattern and presence of acidic protons, with context from taskInfo."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = "Sub-task 4: Integrate the spectral data analyses from Subtasks 1, 2, and 3 to deduce the most probable functional groups and substitution pattern in the unknown compound."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = "Sub-task 5: Compare the deduced structural features from Subtask 4 with the given candidate compounds and select the most reasonable structural suggestion for the unidentified drug."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query", results4['thinking'], results4['answer']],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
