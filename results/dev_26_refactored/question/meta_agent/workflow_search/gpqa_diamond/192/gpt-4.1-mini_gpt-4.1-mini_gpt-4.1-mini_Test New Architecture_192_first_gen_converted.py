async def forward_192(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Apply the transformation plx ∝ 1/r to rewrite the given star count relation 1/(plx^5) in terms of distance r, "
        "explaining the algebraic steps and resulting expression."
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
        "Sub-task 2: Based on the output from Sub-task 1, combine the transformed relation with the differential relationship between parallax and distance to express the number of stars per unit distance interval (dN/dr) as a function of r, "
        "considering all algebraic steps and possible cases."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent expression for the number of stars per unit distance interval (dN/dr) as a function of r."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
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
        "Sub-task 3: Given the derived expression for the number of stars per unit distance interval (dN/dr) as a function of r from Sub-task 2, "
        "debate and select the correct power-law dependence (~ r^n) from the given choices: '~ r^2', '~ r^4', '~ r^3', '~ r^5'."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Select the correct power-law dependence (~ r^n) for the number of stars per unit distance interval based on the derived expression."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
