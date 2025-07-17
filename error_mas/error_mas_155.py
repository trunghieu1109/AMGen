async def forward_155(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze the stereochemical nature of the starting materials (E)- and (Z)-oct-4-ene "
        "and the stereochemical outcome of their epoxidation with mCPBA followed by aqueous acid, "
        "with context from the user query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="stage_0.subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, classify the stereoisomeric products formed from each reaction "
        "in terms of chirality, enantiomeric pairs, and diastereomers, including whether meso forms are possible, "
        "using self-consistency chain-of-thought with context from previous outputs and user query."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Determine the total number and types of stereoisomers present in the combined product mixture "
        "from both reactions, based on outputs from stage_0.subtask_2, using debate agent."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
    }
    results3, log3 = await self.debate(
        subtask_id="stage_1.subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    cot_sc_instruction4 = (
        "Sub-task 4: Predict the chromatographic behavior of the combined mixture on a standard (achiral) reverse-phase HPLC column, "
        "considering co-elution of enantiomers and separation of diastereomers, based on outputs from stage_1.subtask_3, "
        "using self-consistency chain-of-thought."
    )
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results4, log4 = await self.sc_cot(
        subtask_id="stage_1.subtask_4",
        cot_agent_desc=cot_sc_desc4,
        n_repeat=self.max_sc
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Predict the chromatographic behavior of the combined mixture on a chiral HPLC column, "
        "considering resolution of enantiomers and diastereomers, and determine the number of peaks observed, "
        "based on outputs from stage_1.subtask_3, using debate agent."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_3", "answer of stage_1.subtask_3"]
    }
    results5, log5 = await self.debate(
        subtask_id="stage_1.subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(
        [results1['thinking'], results2['thinking'], results3['thinking'], results4['thinking'], results5['thinking']],
        [results1['answer'], results2['answer'], results3['answer'], results4['answer'], results5['answer']]
    )

    return final_answer, logs
