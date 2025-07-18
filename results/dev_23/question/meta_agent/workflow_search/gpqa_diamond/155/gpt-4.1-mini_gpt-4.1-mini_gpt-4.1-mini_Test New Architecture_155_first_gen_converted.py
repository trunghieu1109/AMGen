async def forward_155(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Determine the stereochemical outcome of epoxidation of (E)-oct-4-ene with mCPBA followed by aqueous acid, "
        "identifying the number and configuration of stereoisomeric epoxide products formed, with detailed reasoning and stereochemical rationale."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Determine the stereochemical outcome of epoxidation of (Z)-oct-4-ene with mCPBA followed by aqueous acid, "
        "identifying the number and configuration of stereoisomeric epoxide products formed, with detailed reasoning and stereochemical rationale."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=cot_agent_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Combine the stereochemical information from Sub-task 1 and Sub-task 2 to enumerate and classify all stereoisomeric epoxide products present in the mixture, "
        "including relationships as enantiomers or diastereomers, with detailed reasoning and multiple self-consistent chains of thought."
    )
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Analyze the chromatographic behavior of the combined stereoisomeric epoxide mixture on a standard (achiral) reverse-phase HPLC column and on a chiral HPLC column, "
        "predicting the number of resolved peaks in each chromatogram under ideal resolution conditions, with detailed reasoning and debate."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
