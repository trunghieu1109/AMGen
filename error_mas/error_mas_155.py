async def forward_155(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Determine the stereochemical outcome of the epoxidation of (E)-oct-4-ene with mCPBA and subsequent acidic workup, identifying the stereoisomers formed."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the stereochemical outcome of (E)-oct-4-ene epoxidation and acidic workup."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Determine the stereochemical outcome of the epoxidation of (Z)-oct-4-ene with mCPBA and subsequent acidic workup, identifying the stereoisomers formed."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the stereochemical outcome of (Z)-oct-4-ene epoxidation and acidic workup."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Combine and integrate the stereoisomeric products from both (E)- and (Z)-oct-4-ene reactions to establish the full set of stereoisomers present in the mixture."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of provided solutions of stereoisomeric outcomes from (E)- and (Z)-oct-4-ene epoxidation and acidic workup."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'critic_instruction': critic_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Analyze the combined stereoisomer mixture to predict the number of peaks observed on a standard (achiral) reverse-phase HPLC column, considering co-elution of enantiomers and separation of diastereomers."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Predict the number of peaks on standard (achiral) reverse-phase HPLC for the combined stereoisomer mixture."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'final_decision_instruction': final_decision_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Analyze the combined stereoisomer mixture to predict the number of peaks observed on a chiral HPLC column, considering the resolution of enantiomers and diastereomers under ideal chromatographic conditions."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Predict the number of peaks on chiral HPLC for the combined stereoisomer mixture."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'final_decision_instruction': final_decision_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'context_desc': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(
        [results4['thinking'], results5['thinking']],
        [results4['answer'], results5['answer']]
    )

    return final_answer, logs
