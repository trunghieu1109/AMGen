async def forward_168(self, taskInfo):
    logs = []
    stage0_results = {"subtask_0": [], "subtask_1": [], "subtask_2": []}

    for iteration in range(3):
        cot_instruction_0_0 = (
            "Sub-task 0: Analyze the original decay process 2A -> 2B + 2E + 2V and summarize the energy spectrum characteristics of E particles, "
            "with context from the given nuclear decay query."
        )
        cot_agent_desc_0_0 = {
            "instruction": cot_instruction_0_0,
            "input": [taskInfo],
            "temperature": 0.5,
            "context_desc": ["user query"]
        }
        results_0_0, log_0_0 = await self.sc_cot(
            subtask_id=f"stage_0_subtask_0_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_0,
            n_repeat=self.max_sc
        )
        logs.append(log_0_0)
        stage0_results["subtask_0"].append(results_0_0)

        cot_instruction_0_1 = (
            "Sub-task 1: Based on the output from Sub-task 0, analyze the variant decay process replacing 2V with one massless M particle and infer how this affects the energy distribution and endpoint of E particles, "
            "with context from the original query and Sub-task 0 reasoning."
        )
        final_decision_instruction_0_1 = (
            "Sub-task 1: Synthesize and choose the most consistent answer for the effect on the E particle energy spectrum in the variant decay."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "final_decision_instruction": final_decision_instruction_0_1,
            "input": [taskInfo, results_0_0["thinking"], results_0_0["answer"]],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_0_1, log_0_1 = await self.sc_cot(
            subtask_id=f"stage_0_subtask_1_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_1,
            n_repeat=self.max_sc
        )
        logs.append(log_0_1)
        stage0_results["subtask_1"].append(results_0_1)

        cot_reflect_instruction_0_2 = (
            "Sub-task 2: Refine and simplify the intermediate analysis to produce a clear, structured reasoning about the expected changes in the E particle energy spectrum, "
            "based on outputs from Sub-tasks 0 and 1."
        )
        critic_instruction_0_2 = (
            "Please review and provide the limitations of provided solutions of the energy spectrum changes in the variant decay."
        )
        cot_reflect_desc_0_2 = {
            "instruction": cot_reflect_instruction_0_2,
            "critic_instruction": critic_instruction_0_2,
            "input": [taskInfo, results_0_0["thinking"], results_0_0["answer"], results_0_1["thinking"], results_0_1["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_0_2, log_0_2 = await self.reflexion(
            subtask_id=f"stage_0_subtask_2_iter_{iteration}",
            reflect_desc=cot_reflect_desc_0_2,
            n_repeat=self.max_round
        )
        logs.append(log_0_2)
        stage0_results["subtask_2"].append(results_0_2)

    cot_agent_instruction_1_0 = (
        "Sub-task 0: Evaluate the given answer choices against the refined reasoning from stage_0 and select the candidate that best matches the expected energy spectrum changes."
    )
    # Collect all refined reasonings from stage_0.subtask_2 iterations
    refined_thinkings = [res["thinking"] for res in stage0_results["subtask_2"]]
    refined_answers = [res["answer"] for res in stage0_results["subtask_2"]]

    cot_agent_desc_1_0 = {
        "instruction": cot_agent_instruction_1_0,
        "input": [taskInfo] + refined_thinkings + refined_answers,
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_1_0, log_1_0 = await self.answer_generate(
        subtask_id="stage_1_subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    cot_instruction_2_0 = (
        "Sub-task 0: Validate the selected candidate answer for correctness, consistency with physical principles, and coherence with the problem statement."
    )
    cot_agent_desc_2_0 = {
        "instruction": cot_instruction_2_0,
        "input": [taskInfo, results_1_0["answer"], results_1_0["thinking"]],
        "temperature": 0.0,
        "context_desc": ["user query", "selected answer", "reasoning"]
    }
    results_2_0, log_2_0 = await self.cot(
        subtask_id="stage_2_subtask_0",
        cot_agent_desc=cot_agent_desc_2_0
    )
    logs.append(log_2_0)

    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Perform reflexion and aggregate feedback to confirm or revise the validation outcome of the selected answer."
    )
    critic_instruction_2_1 = (
        "Please review the validation reasoning and provide any corrections or confirmations regarding the selected answer's correctness and consistency."
    )
    cot_reflect_desc_2_1 = {
        "instruction": cot_reflect_instruction_2_1,
        "critic_instruction": critic_instruction_2_1,
        "input": [taskInfo, results_1_0["answer"], results_1_0["thinking"], results_2_0["thinking"], results_2_0["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "selected answer", "initial validation", "validation reasoning", "validation answer"]
    }
    results_2_1, log_2_1 = await self.reflexion(
        subtask_id="stage_2_subtask_1",
        reflect_desc=cot_reflect_desc_2_1,
        n_repeat=self.max_round
    )
    logs.append(log_2_1)

    aggregate_instruction_2_2 = (
        "Sub-task 2: Produce a final assessment and justification for the selected answer choice based on validation and reflexion results."
    )
    aggregate_desc_2_2 = {
        "instruction": aggregate_instruction_2_2,
        "input": [taskInfo, results_1_0["answer"], results_1_0["thinking"], results_2_0["answer"], results_2_0["thinking"], results_2_1["answer"], results_2_1["thinking"]],
        "temperature": 0.0,
        "context_desc": ["user query", "selected answer", "initial validation", "validation reasoning", "reflexion"]
    }
    results_2_2, log_2_2 = await self.aggregate(
        subtask_id="stage_2_subtask_2",
        aggregate_desc=aggregate_desc_2_2
    )
    logs.append(log_2_2)

    final_answer = await self.make_final_answer(results_2_2["thinking"], results_1_0["answer"])

    return final_answer, logs
