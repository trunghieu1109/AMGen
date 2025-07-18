async def forward_192(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Clarify and explicitly determine whether the given relation 'number of stars varies with parallax as 1/(plx^5)' refers to a cumulative count N(plx) or a differential distribution dN/d(plx). "
        "If ambiguous, analyze both interpretations and their implications for subsequent steps, with context from the user query."
    )
    debate_desc1 = {
        "instruction": cot_instruction1,
        "final_decision_instruction": "Sub-task 1: Clarify the interpretation of the given relation for number of stars vs parallax.",
        "input": [taskInfo],
        "context_desc": ["user query"],
        "temperature": 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, express the given differential distribution dN/d(plx) proportional to 1/(plx^5) in terms of distance r, using plx proportional to 1/r. "
        "Rewrite the distribution explicitly as a function of r, ensuring correct interpretation from Subtask 1."
    )
    final_decision_instruction2 = "Sub-task 2: Synthesize and choose the most consistent expression of dN/d(plx) in terms of r."
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_sc_instruction3 = (
        "Sub-task 3: Apply the chain rule and Jacobian transformation to convert the differential distribution from parallax to distance: compute dN/dr = (dN/dplx) * |d(plx)/dr|. "
        "Derive the expression for dN/dr as a function of r, ensuring all derivatives and absolute values are correctly handled, using the expression from Subtask 2."
    )
    final_decision_instruction3 = "Sub-task 3: Synthesize and confirm the correct expression for dN/dr as a function of r."
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
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

    debate_instruction4 = (
        "Sub-task 4: Based on the derived expression for dN/dr from Subtask 3, select the correct power-law dependence (~ r^n) for the number of stars per unit distance interval from the given choices: ~ r^2, ~ r^4, ~ r^3, ~ r^5. "
        "Verify the physical and mathematical consistency of the final result and reflect on the correctness of previous steps."
    )
    final_decision_instruction4 = "Sub-task 4: Select the correct power-law dependence for the number of stars per unit distance interval."
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
