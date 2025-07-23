async def forward_177(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given information from the query, including the interaction Lagrangian, definitions of fields and operators, and the problem requirements."
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
        "Sub-task 2: Identify and assign the canonical mass dimensions of the fields and operators involved (psi, bar{psi}, F^{mu nu}, sigma_{mu nu}), and the overall dimension of the Lagrangian density in four-dimensional spacetime, based on the output from Sub-task 1."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent assignment of mass dimensions for the fields and operators."
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

    cot_sc_instruction3 = (
        "Sub-task 3: Construct the intermediate representation by calculating the total mass dimension of the interaction term kappa bar{psi} sigma_{mu nu} psi F^{mu nu} and deduce the mass dimension of the coupling constant kappa, based on the output from Sub-task 2."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Synthesize and choose the most consistent mass dimension of kappa."
    )
    cot_sc_desc3 = {
        "instruction": cot_sc_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2["thinking"], results2["answer"]],
        "temperature": 0.5,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Derive the renormalizability status of the theory based on the mass dimension of kappa, applying standard QFT criteria for renormalizability, using the output from Sub-task 3."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Decide if the theory is renormalizable or not based on the mass dimension of kappa."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_reflect_instruction5 = (
        "Sub-task 5: Validate the derived mass dimension and renormalizability conclusion against the provided multiple-choice options and select the correct answer, using outputs from Sub-tasks 3 and 4."
    )
    critic_instruction5 = (
        "Please review and provide the limitations of provided solutions and select the best matching multiple-choice option."
    )
    cot_reflect_desc5 = {
        "instruction": cot_reflect_instruction5,
        "critic_instruction": critic_instruction5,
        "input": [taskInfo, results3["thinking"], results3["answer"], results4["thinking"], results4["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.reflexion(
        subtask_id="subtask_5",
        reflect_desc=cot_reflect_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
