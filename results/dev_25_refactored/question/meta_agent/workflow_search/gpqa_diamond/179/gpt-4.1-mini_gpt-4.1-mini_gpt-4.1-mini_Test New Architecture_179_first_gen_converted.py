async def forward_179(self, taskInfo):
    logs = []

    cot_sc_instruction0_1 = (
        "Sub-task 1: Extract and transform all given physical parameters and constants "
        "(charge magnitude, distances, Coulomb constant, elementary charge) into usable numerical values for calculations, "
        "with context from the user query."
    )
    cot_sc_desc0_1 = {
        'instruction': cot_sc_instruction0_1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent set of physical parameters.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results0_1, log0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc0_1,
        n_repeat=self.max_sc
    )
    logs.append(log0_1)

    cot_sc_instruction1_1 = (
        "Sub-task 1: Model the electrostatic system by defining the configuration of charges: "
        "one fixed at point P and 12 charges constrained on a sphere of radius 2 m, and formulate expressions "
        "for all pairwise electrostatic potential energies, using the extracted parameters from stage_0.subtask_1."
    )
    cot_sc_desc1_1 = {
        'instruction': cot_sc_instruction1_1,
        'final_decision_instruction': "Sub-task 1: Synthesize and choose the most consistent electrostatic model and energy expressions.",
        'input': [taskInfo, results0_1['thinking'], results0_1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1_1,
        n_repeat=self.max_sc
    )
    logs.append(log1_1)

    debate_instruction1_2 = (
        "Sub-task 2: Determine the minimum energy configuration of the 12 charges on the sphere (related to the Thomson problem) "
        "and calculate the total electrostatic potential energy of the system including interactions between the center charge and the 12 charges "
        "as well as among the 12 charges themselves, based on the model and parameters from previous subtasks."
    )
    final_decision_instruction1_2 = "Sub-task 2: Select the most plausible minimum energy value and configuration for the system."
    debate_desc1_2 = {
        'instruction': debate_instruction1_2,
        'final_decision_instruction': final_decision_instruction1_2,
        'input': [taskInfo, results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']],
        'context_desc': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"],
        'temperature': 0.5
    }
    results1_2, log1_2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc1_2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    debate_instruction2_1 = (
        "Sub-task 1: Evaluate the computed minimum energy against the provided answer choices and select the correct minimum energy value "
        "rounded to three decimals in Joules, based on the calculated results from stage_1.subtask_2."
    )
    final_decision_instruction2_1 = "Sub-task 1: Choose the best matching answer choice for the minimum energy of the system."
    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'final_decision_instruction': final_decision_instruction2_1,
        'input': [taskInfo, results1_2['thinking'], results1_2['answer']],
        'context_desc': ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        'temperature': 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1['thinking'], results2_1['answer'])
    return final_answer, logs
