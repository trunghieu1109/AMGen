async def forward_161(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the given metric, domain, and problem parameters, "
        "including the definition of the pseudosphere radius and the provided answer choices."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2_1 = (
        "Sub-task 1: Rewrite the metric ds^2 = 32/(4 - x^2 - y^2)(dx^2 + dy^2) in polar coordinates r, theta, "
        "and set up the integral expression for the area of the pseudosphere of radius r=2."
    )
    cot_sc_instruction2_2 = (
        "Sub-task 2: Evaluate the area integral over the domain r in [0,2), analyze the behavior near the boundary r=2, "
        "and determine whether the area is finite or infinite."
    )
    final_decision_instruction2_2 = (
        "Sub-task 2: Synthesize and choose the most consistent conclusion about the finiteness of the area."
    )

    cot_sc_desc2_1 = {
        "instruction": cot_sc_instruction2_1,
        "final_decision_instruction": None,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }

    cot_sc_desc2_2 = {
        "instruction": cot_sc_instruction2_2,
        "final_decision_instruction": final_decision_instruction2_2,
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }

    results2_1, log2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2_1,
        n_repeat=self.max_sc
    )
    logs.append(log2_1)

    results2_2, log2_2 = await self.sc_cot(
        subtask_id="stage_2.subtask_2",
        cot_agent_desc=cot_sc_desc2_2,
        n_repeat=self.max_sc
    )
    logs.append(log2_2)

    debate_instruction3 = (
        "Sub-task 1: Compare the computed area result with the given answer choices and select the best matching candidate."
    )
    final_decision_instruction3 = (
        "Sub-task 1: Provide the final answer choice that best matches the computed area of the pseudosphere."
    )

    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2_2["thinking"], results2_2["answer"]],
        "context_desc": ["user query", "thinking of stage_2.subtask_2", "answer of stage_2.subtask_2"],
        "temperature": 0.5
    }

    results3, log3 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3["thinking"], results3["answer"])
    return final_answer, logs
