async def forward_155(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Determine the detailed stereochemical outcome of the epoxidation of (E)-oct-4-ene with mCPBA followed by aqueous acid-catalyzed ring opening. "
        "Explicitly enumerate all possible stereoisomeric diol products formed, including their configurations and relationships as enantiomers or diastereomers. "
        "Avoid stopping at the epoxide stage and consider stereochemical consequences of acid-catalyzed ring opening on the (E)-alkene-derived epoxide."
    )
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Determine the detailed stereochemical outcome of the epoxidation of (Z)-oct-4-ene with mCPBA followed by aqueous acid-catalyzed ring opening. "
        "Explicitly enumerate all possible stereoisomeric diol products formed, including all diastereomers and enantiomers, considering regioselectivity and stereoselectivity of the acid-catalyzed ring opening step. "
        "Address previous oversimplifications and provide a comprehensive stereochemical analysis of the (Z)-alkene-derived products."
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
        "Sub-task 3: Combine the stereochemical information from subtasks 1 and 2 to enumerate and classify all stereoisomeric diol products present in the combined reaction mixture. "
        "Clearly identify the total number of stereoisomers, their relationships as enantiomers or diastereomers, and prepare this comprehensive stereochemical set for chromatographic analysis. "
        "Ensure the final species considered are the ring-opened diols, not the epoxides."
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
        "Sub-task 4: Analyze the chromatographic behavior of the combined stereoisomeric diol mixture on both a standard (achiral) reverse-phase HPLC column and a chiral HPLC column. "
        "Predict the number of resolved peaks in each chromatogram under ideal chromatographic resolution conditions. "
        "Use the complete and correct stereoisomeric diol set from subtask 3 and incorporate known chromatographic principles: enantiomers co-elute on achiral columns but can be resolved on chiral columns, while diastereomers can be separated on achiral columns. "
        "Focus on the final diol products."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results3['thinking'], results3['answer']],
        'input': [taskInfo, results3['thinking'], results3['answer']],
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
