async def forward_161(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Analyze and transform the given metric ds^2 = 32/(4 - x^2 - y^2)*(dx^2 + dy^2) "
        "into a form suitable for area calculation, including expressing the area element in polar coordinates and identifying the domain of integration. "
        "Provide detailed Chain-of-Thought reasoning."
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
        "Sub-task 2: Based on the output from Sub-task 1, compute the area of the pseudosphere by integrating the derived area element "
        "over the disk of radius r=2. Use Self-Consistency Chain-of-Thought to consider multiple reasoning paths and synthesize the most consistent result."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent computed area for the pseudosphere of radius 2."
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
        "Sub-task 3: Evaluate the computed area result from Sub-task 2 against the provided answer choices: '+infinity', "
        "'4pi(x^2 + y^2)', '0', and '4pi(x^2 - y^2)'. Debate the correctness and select the best matching answer."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the correct answer choice for the area of the pseudosphere of radius 2 based on the computed area."
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
