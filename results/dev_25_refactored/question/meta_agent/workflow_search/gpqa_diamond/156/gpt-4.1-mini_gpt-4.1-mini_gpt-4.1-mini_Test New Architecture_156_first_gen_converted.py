async def forward_156(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Identify the viral agent causing the outbreak by determining its genetic material type and sequence (DNA or RNA), considering retrovirus characteristics, with context from the query."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, determine the appropriate molecular target for detection (viral genome via nucleic acid or host antibodies such as IgG) based on infection stage and retrovirus biology, with context from the query and previous results."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent molecular target for detection based on previous analysis."
    )
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

    cot_reflect_instruction3 = (
        "Sub-task 3: Design the molecular diagnostic assay (e.g., PCR variants or ELISA) tailored to the identified molecular target, ensuring quick and accurate detection, based on previous subtasks."
    )
    critic_instruction3 = (
        "Please review and provide the limitations of the proposed diagnostic assay designs considering speed, accuracy, and feasibility."
    )
    cot_reflect_desc3 = {
        "instruction": cot_reflect_instruction3,
        "critic_instruction": critic_instruction3,
        "input": [taskInfo, results1["thinking"], results1["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate and select the optimal diagnostic kit design by comparing sensitivity, specificity, speed, and feasibility of PCR-based versus antibody-based methods, based on previous assay design outputs."
    )
    final_decision_instruction4 = (
        "Sub-task 4: Select the best diagnostic kit design considering all evaluation criteria."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context_desc": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    cot_sc_instruction5 = (
        "Sub-task 5: Integrate the selected diagnostic approach into a practical kit design, outlining key components and workflow for rapid deployment in the outbreak setting, based on the selected design from Sub-task 4."
    )
    final_decision_instruction5 = (
        "Sub-task 5: Provide a detailed practical kit design and workflow for rapid deployment."
    )
    cot_sc_desc5 = {
        "instruction": cot_sc_instruction5,
        "final_decision_instruction": final_decision_instruction5,
        "input": [taskInfo, results4["thinking"], results4["answer"]],
        "temperature": 0.5,
        "context_desc": ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5, log5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5["thinking"], results5["answer"])
    return final_answer, logs
