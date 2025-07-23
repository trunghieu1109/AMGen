async def forward_183(self, taskInfo):
    logs = []

    cot_instruction_stage0 = (
        "Sub-task 1: Analyze each provided reaction sequence stepwise to identify intermediate transformations and their chemical rationale. "
        "Input: taskInfo containing the question and all four choices."
    )
    cot_agent_desc_stage0 = {
        "instruction": cot_instruction_stage0,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_stage0, log_stage0 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_stage0
    )
    logs.append(log_stage0)

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []},
        "stage_1.subtask_3": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_reflect_instruction = (
            "Sub-task 1: Simplify and consolidate intermediate analyses to highlight key reaction order and regioselectivity considerations. "
            "Input: taskInfo and all previous thinking and answers from stage_0.subtask_1 and all previous iterations of stage_1 subtasks."
        )
        critic_instruction = (
            "Please review and provide the limitations of provided solutions of synthetic route analyses and refinements."
        )
        inputs_reflexion = [taskInfo, results_stage0['thinking'], results_stage0['answer']]
        for subtask_id in ['stage_1.subtask_1', 'stage_1.subtask_2', 'stage_1.subtask_3']:
            inputs_reflexion += loop_results[subtask_id]['thinking'] + loop_results[subtask_id]['answer']

        cot_reflect_desc = {
            "instruction": cot_reflect_instruction,
            "critic_instruction": critic_instruction,
            "input": inputs_reflexion,
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"] +
                            ["thinking and answer of previous iterations of stage_1 subtasks"]
        }
        results_reflexion, log_reflexion = await self.reflexion(
            subtask_id="stage_1.subtask_1",
            reflect_desc=cot_reflect_desc,
            n_repeat=1
        )
        logs.append(log_reflexion)
        loop_results["stage_1.subtask_1"]["thinking"].append(results_reflexion['thinking'])
        loop_results["stage_1.subtask_1"]["answer"].append(results_reflexion['answer'])

        aggregate_instruction = (
            "Sub-task 2: Evaluate each candidate sequence against criteria for high yield and correct substitution pattern. "
            "Input: taskInfo and all thinking and answers from stage_0.subtask_1 and all previous iterations of stage_1 subtasks."
        )
        inputs_aggregate = [taskInfo, results_stage0['thinking'], results_stage0['answer']]
        for subtask_id in ['stage_1.subtask_1']:
            inputs_aggregate += loop_results[subtask_id]['thinking'] + loop_results[subtask_id]['answer']

        aggregate_desc = {
            "instruction": aggregate_instruction,
            "input": inputs_aggregate,
            "temperature": 0.0,
            "context": ["user query", "solutions generated from stage_0.subtask_1 and stage_1.subtask_1"]
        }
        results_aggregate, log_aggregate = await self.aggregate(
            subtask_id="stage_1.subtask_2",
            aggregate_desc=aggregate_desc
        )
        logs.append(log_aggregate)
        loop_results["stage_1.subtask_2"]["thinking"].append(results_aggregate['thinking'])
        loop_results["stage_1.subtask_2"]["answer"].append(results_aggregate['answer'])

        answer_generate_instruction = (
            "Sub-task 3: Select the best candidate synthetic route based on refined analyses and evaluations. "
            "Input: taskInfo and all thinking and answers from stage_1.subtask_1 and stage_1.subtask_2 from all iterations."
        )
        inputs_answer_generate = [taskInfo]
        for subtask_id in ['stage_1.subtask_1', 'stage_1.subtask_2']:
            inputs_answer_generate += loop_results[subtask_id]['thinking'] + loop_results[subtask_id]['answer']

        cot_agent_desc_answer_generate = {
            "instruction": answer_generate_instruction,
            "input": inputs_answer_generate,
            "temperature": 0.0,
            "context": ["user query", "thinking and answer of stage_1.subtask_1 and stage_1.subtask_2"]
        }
        results_answer_generate, log_answer_generate = await self.answer_generate(
            subtask_id="stage_1.subtask_3",
            cot_agent_desc=cot_agent_desc_answer_generate
        )
        logs.append(log_answer_generate)
        loop_results["stage_1.subtask_3"]["thinking"].append(results_answer_generate['thinking'])
        loop_results["stage_1.subtask_3"]["answer"].append(results_answer_generate['answer'])

    review_instruction = (
        "Sub-task 1: Apply systematic criteria to assess the correctness and feasibility of the selected synthetic route. "
        "Input: taskInfo and the final selected synthetic route thinking and answer from stage_1.subtask_3."
    )
    inputs_review = [taskInfo, loop_results["stage_1.subtask_3"]["thinking"][-1], loop_results["stage_1.subtask_3"]["answer"][-1]]
    review_desc = {
        "instruction": review_instruction,
        "input": inputs_review,
        "temperature": 0.0,
        "context": ["user query", "thinking and answer of stage_1.subtask_3"]
    }
    results_review, log_review = await self.review(
        subtask_id="stage_2.subtask_1",
        review_desc=review_desc
    )
    logs.append(log_review)

    final_answer = await self.make_final_answer(loop_results["stage_1.subtask_3"]["thinking"][-1], loop_results["stage_1.subtask_3"]["answer"][-1])
    return final_answer, logs
