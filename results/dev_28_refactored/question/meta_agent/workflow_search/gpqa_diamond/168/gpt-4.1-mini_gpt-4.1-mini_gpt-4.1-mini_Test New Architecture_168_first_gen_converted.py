async def forward_168(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Analyze and characterize the kinematic and energy distribution relationships among the particles in the original decay 2A -> 2B + 2E + 2V, focusing on why the E particle energy spectrum is continuous with endpoint Q."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent explanation for the continuous energy spectrum of E particles in the original decay.",
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Analyze the variant decay 2A -> 2B + 2E + M, focusing on how replacing two V particles with one massless M affects the energy partitioning, spectrum continuity, and endpoint of the E particles' energy distribution."
    )
    debate_desc2 = {
        "instruction": debate_instruction2,
        "final_decision_instruction": "Sub-task 2: Debate and conclude the impact of replacing 2V with 1 massless M on the E particle energy spectrum.",
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "temperature": 0.5
    }
    results2, log2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Derive the expected shape and endpoint changes of the E particle energy spectrum in the variant decay by applying multi-body kinematics and energy conservation principles, comparing quantitatively to the original decay spectrum."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": "Sub-task 3: Synthesize and select the most consistent quantitative derivation of the spectrum changes.",
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    cot_reflect_instruction4 = (
        "Sub-task 4: Validate the derived conclusions about the spectrum continuity and endpoint changes against known physical principles and constraints, ensuring consistency and correctness."
    )
    critic_instruction4 = (
        "Please review and provide the limitations of provided solutions regarding the spectrum continuity and endpoint changes in the variant decay."
    )
    cot_reflect_desc4 = {
        "instruction": cot_reflect_instruction4,
        "critic_instruction": critic_instruction4,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"], results3["thinking"], results3["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4, log4 = await self.reflexion(
        subtask_id="subtask_4",
        reflect_desc=cot_reflect_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Evaluate all candidate answer choices against the validated conclusions and select the best matching option describing how the E particle energy spectrum changes in the variant decay."
    )
    debate_desc5 = {
        "instruction": debate_instruction5,
        "final_decision_instruction": "Sub-task 5: Select the best answer choice describing the change in the E particle energy spectrum in the variant decay.",
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "context_desc": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "temperature": 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
