async def forward_155(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Perform a detailed stereochemical analysis of the starting materials ((E)- and (Z)-oct-4-ene) and reagents (mCPBA, aqueous acid). "
        "Explicitly list substituents on each alkene carbon, determine if substituents are identical or different, and apply the stereochemical rule: cis-alkenes with identical substituents yield meso epoxides, trans-alkenes yield racemic epoxides. "
        "Clarify the stereochemical outcome of epoxidation and subsequent aqueous acid treatment, including whether the products are epoxides or diols and their stereochemical configurations. "
        "Avoid the previous error of assuming both alkenes yield racemic mixtures and explicitly identify meso vs racemic products."
    )
    debate_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"],
        'role': self.debate_role
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, enumerate and classify all stereoisomeric products formed from each reaction. "
        "Explicitly identify the number of stereoisomers, distinguish enantiomers from meso forms, and classify diastereomeric relationships between products from (E)- and (Z)-oct-4-ene. "
        "Address the previous failure of miscounting stereoisomers by carefully mapping stereochemical relationships and ensuring accurate stereoisomer counts for downstream chromatographic predictions."
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

    debate_instruction3 = (
        "Sub-task 3: Predict the chromatographic behavior of the combined product mixture on a standard (achiral) reverse-phase HPLC column. "
        "Use the stereochemical classification from Subtask 2 to determine which stereoisomers are enantiomers (expected to co-elute) and which are diastereomers (expected to separate). "
        "Explicitly consider the impact of meso compounds on peak count and avoid the previous oversimplification that led to incorrect peak numbers. "
        "Provide a clear rationale for the expected number of resolved peaks on the achiral column."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'role': self.debate_role
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Predict the chromatographic behavior of the combined product mixture on a chiral HPLC column. "
        "Based on Subtask 2's stereochemical mapping, determine how enantiomers are resolved on the chiral column, including meso forms that are achiral and thus not resolved into enantiomers. "
        "Explicitly address the previous failure to consider meso compounds and their effect on peak numbers. "
        "Provide a detailed explanation of the expected number of peaks on the chiral column."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'role': self.debate_role
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Integrate the chromatographic predictions from Subtasks 3 and 4 to determine the final expected number of peaks observed in each chromatogram. "
        "Select the correct answer choice accordingly. "
        "Explicitly reconcile the stereochemical and chromatographic analyses, ensuring consistency and avoiding previous errors of miscounting or misclassification of peaks."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
