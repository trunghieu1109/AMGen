async def forward_182(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Perform a detailed structural analysis of the starting compound 2-formyl-5-vinylcyclohex-3-enecarboxylic acid: "
        "explicitly determine and confirm the exact molecular formula (count of C, H, O), the number of rings, and the initial Index of Hydrogen Deficiency (IHD) numerically. "
        "This step addresses previous errors caused by incorrect ring count assumptions and lack of numerical IHD calculation, ensuring a solid structural foundation for subsequent reasoning."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Analyze and confirm the chemical reactivity and scope of red phosphorus and excess HI on the functional groups present in the starting compound "
        "(formyl, vinyl, carboxylic acid) and on the ring double bond. Explicitly verify which functional groups are reduced or transformed and which unsaturations remain intact, citing standard reaction mechanisms and literature. "
        "This subtask prevents the mechanistic error of assuming HI/P_red hydrogenates all double bonds, a key failure in previous attempts."
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
        "Sub-task 3: Derive the most likely product structure after reaction with red phosphorus and excess HI by applying the confirmed transformations from subtask_2 "
        "to the starting compound's structure from subtask_1. Explicitly document changes to functional groups and unsaturations, ensuring no assumptions contradict the verified reaction scope. "
        "This step avoids propagation of incorrect product structures."
    )
    critic_instruction3 = (
        "Please review the derived product structure filtering and provide its limitations."
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
        "Sub-task 4: Critically verify the derived product structure by cross-checking the number of rings and remaining unsaturations (pi bonds) "
        "against the confirmed reaction mechanism and structural analysis. This verification step is introduced to catch and correct any overlooked errors in ring count or unsaturation before calculating the product's IHD, "
        "addressing previous failures where ring count was overestimated and assumptions went unchallenged."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results3['thinking'], results3['answer'], results2['thinking'], results2['answer'], results1['thinking'], results1['answer']],
        'input': [taskInfo, results3['thinking'], results3['answer'], results2['thinking'], results2['answer'], results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Calculate the Index of Hydrogen Deficiency (IHD) of the verified product structure based on the confirmed number of rings and pi bonds remaining after the reaction. "
        "This calculation must be consistent with the verified structure and mechanistic understanding to avoid errors in final IHD determination."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    debate_instruction6 = (
        "Sub-task 6: Select the correct IHD value of the product from the given choices (0, 1, 3, 5) based on the computed IHD. "
        "Conduct a debate among agents to critically evaluate and confirm the final answer, ensuring that all previous assumptions and calculations are consistent and justified, "
        "thus preventing consensus on incorrect conclusions as happened previously."
    )
    debate_desc6 = {
        'instruction': debate_instruction6,
        'context': ["user query", results5['thinking'], results5['answer']],
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results6, log6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc6,
        n_repeat=self.max_round
    )
    logs.append(log6)

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'])
    return final_answer, logs
