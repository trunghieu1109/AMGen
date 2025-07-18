async def forward_172(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction0_1 = (
        "Sub-task 1: Extract and convert all given physical quantities into consistent SI units and identify constants needed for calculations (electron mass, reduced Planck constant ħ)."
    )
    cot_agent_desc0_1 = {
        "instruction": cot_instruction0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results0_1, log0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc0_1,
        n_repeat=self.max_sc
    )
    logs.append(log0_1)

    cot_instruction1_1 = (
        "Sub-task 1: Apply the Heisenberg uncertainty principle to calculate the minimum uncertainty in momentum Δp from the given position uncertainty Δx."
    )
    cot_agent_desc1_1 = {
        "instruction": cot_instruction1_1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent answer for minimum uncertainty in momentum Δp.",
        "input": [taskInfo, results0_1['thinking'], results0_1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1_1,
        n_repeat=self.max_sc
    )
    logs.append(log1_1)

    cot_reflect_instruction1_2 = (
        "Sub-task 2: Relate the uncertainty in momentum Δp to the uncertainty in kinetic energy ΔE using the classical kinetic energy formula and the electron's velocity."
    )
    critic_instruction1_2 = (
        "Please review and provide the limitations of provided solutions relating Δp to ΔE, considering assumptions and approximations."
    )
    cot_reflect_desc1_2 = {
        "instruction": cot_reflect_instruction1_2,
        "critic_instruction": critic_instruction1_2,
        "input": [taskInfo, results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results1_2, log1_2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc1_2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    debate_instruction2_1 = (
        "Sub-task 1: Evaluate the calculated minimum uncertainty in energy ΔE against the provided multiple-choice options and select the closest matching value."
    )
    final_decision_instruction2_1 = (
        "Sub-task 1: Select the best matching multiple-choice answer for minimum uncertainty in energy ΔE."
    )
    debate_desc2_1 = {
        "instruction": debate_instruction2_1,
        "final_decision_instruction": final_decision_instruction2_1,
        "input": [taskInfo, results1_2['thinking'], results1_2['answer']],
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1['thinking'], results2_1['answer'])
    return final_answer, logs
