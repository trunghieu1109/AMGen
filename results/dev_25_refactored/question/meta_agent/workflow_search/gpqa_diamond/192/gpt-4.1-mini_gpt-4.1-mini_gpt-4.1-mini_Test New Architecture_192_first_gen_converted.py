async def forward_192(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Apply the transformation from parallax (plx) to distance (r) using the inverse relationship plx = 1/r, "
        "and express the number of stars as a function of distance."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Combine the transformed expression with the Jacobian of the variable change to find the number of stars per unit distance interval, "
        "accounting for the differential relationship between parallax and distance."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent expression for the number of stars per unit distance interval."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the resulting expression for the number of stars per unit distance and select the correct answer choice (~ r^n) that matches the derived power law."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the correct power law dependence of the number of stars per unit distance from the given choices."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3["thinking"], results3["answer"])
    return final_answer, logs
