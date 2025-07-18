async def forward_153(self, taskInfo):
    logs = []

    cot_sc_instruction0 = "Sub-task 1: Extract and summarize quantitative spectral data from the mass spectrum, IR spectrum, and 1H NMR data, including molecular ion peaks, isotopic patterns, characteristic IR bands, and NMR chemical shifts and splitting patterns, with context from the provided query."
    final_decision_instruction0 = "Sub-task 1: Synthesize and choose the most consistent and accurate summary of the spectral data extracted."
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'final_decision_instruction': final_decision_instruction0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results0, log0 = await self.sc_cot(
        subtask_id="stage0_subtask_1",
        cot_agent_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    logs.append(log0)

    debate_instruction1_1 = (
        "Sub-task 1: Analyze the extracted spectral data to identify key structural features such as presence of chlorine (from isotopic pattern), "
        "functional groups (carboxylic acid vs aldehyde from IR and NMR), and substitution pattern on the aromatic ring (from NMR splitting and integration), "
        "using the summary from stage 0."
    )
    final_decision_instruction1_1 = (
        "Sub-task 1: Debate and decide on the key structural features identified from the spectral data summary."
    )
    debate_desc1_1 = {
        'instruction': debate_instruction1_1,
        'final_decision_instruction': final_decision_instruction1_1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'context_desc': ["user query", "thinking of stage0_subtask_1", "answer of stage0_subtask_1"],
        'temperature': 0.5
    }
    results1_1, log1_1 = await self.debate(
        subtask_id="stage1_subtask_1",
        debate_desc=debate_desc1_1,
        n_repeat=self.max_round
    )
    logs.append(log1_1)

    cot_sc_instruction1_2 = (
        "Sub-task 2: Classify the spectral features to narrow down the possible structural motifs and substitution patterns consistent with the data, "
        "focusing on differentiating positional isomers and functional groups among the candidate structures, using the outputs from stage 0 and stage 1 subtask 1."
    )
    final_decision_instruction1_2 = (
        "Sub-task 2: Synthesize and choose the most consistent classification of spectral features to narrow down candidate structures."
    )
    cot_sc_desc1_2 = {
        'instruction': cot_sc_instruction1_2,
        'final_decision_instruction': final_decision_instruction1_2,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1_1['thinking'], results1_1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage0_subtask_1", "answer of stage0_subtask_1", "thinking of stage1_subtask_1", "answer of stage1_subtask_1"]
    }
    results1_2, log1_2 = await self.sc_cot(
        subtask_id="stage1_subtask_2",
        cot_agent_desc=cot_sc_desc1_2,
        n_repeat=self.max_sc
    )
    logs.append(log1_2)

    debate_instruction2_1 = (
        "Sub-task 1: Evaluate and prioritize the candidate structures (4-chlorobenzoic acid, 2-chlorobenzoic acid, 3-chloro-2-hydroxybenzaldehyde, phenyl chloroformate) "
        "against the interpreted spectral data from previous subtasks to select the most reasonable structural suggestion for the unidentified compound."
    )
    final_decision_instruction2_1 = (
        "Sub-task 1: Debate and decide on the best candidate structure based on all previous analyses."
    )
    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'final_decision_instruction': final_decision_instruction2_1,
        'input': [taskInfo, results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'context_desc': ["user query", "thinking of stage1_subtask_1", "answer of stage1_subtask_1", "thinking of stage1_subtask_2", "answer of stage1_subtask_2"],
        'temperature': 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage2_subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1['thinking'], results2_1['answer'])
    return final_answer, logs
