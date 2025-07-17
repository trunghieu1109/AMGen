async def forward_153(self, taskInfo):
    logs = []

    cot_instruction0 = "Sub-task 1: Extract and summarize the key spectral data and physical properties from the input, including mass spectrometry peaks, IR absorption bands, and 1H NMR chemical shifts and splitting patterns."
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results0, log0 = await self.debate(
        subtask_id="stage_0.subtask_1",
        debate_desc=cot_agent_desc0,
        n_repeat=self.max_round
    )
    logs.append(log0)

    cot_instruction1_1 = "Sub-task 1: Analyze the mass spectrometry data to identify isotopic patterns and infer the presence of specific elements (e.g., chlorine) and molecular weight, using the summarized spectral data from stage_0.subtask_1."
    cot_agent_desc1_1 = {
        'instruction': cot_instruction1_1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_1, log1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=cot_agent_desc1_1,
        n_repeat=self.max_round
    )
    logs.append(log1_1)

    cot_instruction1_2 = "Sub-task 2: Interpret the IR spectrum to identify functional groups, focusing on the broad O-H stretch and sharp carbonyl peak, and correlate with possible compound classes, using the summarized spectral data from stage_0.subtask_1."
    cot_agent_desc1_2 = {
        'instruction': cot_instruction1_2,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_2, log1_2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=cot_agent_desc1_2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    cot_instruction1_3 = "Sub-task 3: Analyze the 1H NMR data to deduce the aromatic substitution pattern and presence of acidic protons, integrating this with the mass and IR data to narrow down structural possibilities, using the summarized spectral data from stage_0.subtask_1."
    cot_agent_desc1_3 = {
        'instruction': cot_instruction1_3,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_3, log1_3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=cot_agent_desc1_3,
        n_repeat=self.max_round
    )
    logs.append(log1_3)

    cot_sc_instruction2 = (
        "Sub-task 1: Integrate the analyzed mass spectrometry, IR, and NMR data to propose the most reasonable chemical structure for the unknown compound, "
        "selecting the best match from the given choices: 4-chlorobenzoic acid, 2-chlorobenzoic acid, 3-Chloro-2-hydroxybenzaldehyde, Phenyl chloroformate. "
        "Use the outputs from stage_1.subtask_1, stage_1.subtask_2, and stage_1.subtask_3 to support your reasoning."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer'], results1_3['thinking'], results1_3['answer']],
        'temperature': 0.5,
        'context': [
            "user query",
            "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2",
            "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"
        ]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    final_answer = await self.make_final_answer(results2['thinking'], results2['answer'])
    return final_answer, logs
