async def forward_155(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and classify the stereochemical nature of the starting materials ((E)- and (Z)-oct-4-ene) and the reagents (mCPBA, aqueous acid), "
        "including the expected stereochemical outcome of the epoxidation and subsequent aqueous acid treatment, with context from taskInfo."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=cot_agent_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, determine the stereochemical identity and number of stereoisomeric products formed from each reaction, "
        "including whether the products are enantiomers or diastereomers, and how the (E)- and (Z)-alkenes differ in product stereochemistry, with context from taskInfo and results of Sub-task 1."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1.get('thinking', ''), results1.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the chromatographic behavior of the combined product mixture on a standard (achiral) reverse-phase HPLC column, "
        "predicting the number of resolved peaks based on stereochemical relationships (enantiomers co-elute, diastereomers separate), with context from taskInfo and results of Sub-task 2."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results2.get('thinking', ''), results2.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the chromatographic behavior of the combined product mixture on a chiral HPLC column, "
        "predicting the number of resolved peaks considering that enantiomers can be separated on chiral columns, with context from taskInfo and results of Sub-task 2."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'input': [taskInfo, results2.get('thinking', ''), results2.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Integrate the chromatographic predictions from Sub-tasks 3 and 4 to determine the final expected number of peaks observed in each chromatogram, "
        "and select the correct answer choice accordingly, with context from taskInfo and results of Sub-tasks 3 and 4."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results3.get('thinking', ''), results3.get('answer', ''), results4.get('thinking', ''), results4.get('answer', '')],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5.get('thinking', ''), results5.get('answer', ''))
    return final_answer, logs
