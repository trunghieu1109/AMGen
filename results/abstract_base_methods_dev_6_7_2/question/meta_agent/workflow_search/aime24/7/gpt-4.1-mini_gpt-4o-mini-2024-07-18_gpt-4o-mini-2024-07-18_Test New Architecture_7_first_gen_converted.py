async def forward_7(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction0 = "Subtask 1: Identify and list the two given logarithmic equations from the problem statement"
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results0['list_thinking']):
        agents.append(f"SC-CoT agent {results0['cot_agent'][idx].id}, identifying equations, thinking: {results0['list_thinking'][idx]}; answer: {results0['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    candidate_relationships = []
    for i in range(2):
        cot_instruction1 = f"Subtask 2.{i+1}: Convert the logarithmic equation {i+1} from logarithmic form to exponential form and derive a relationship between x and y"
        cot_agent_desc1 = {
            'instruction': cot_instruction1,
            'input': [taskInfo, results0['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "identified equations"]
        }
        results1 = await self.cot(
            subtask_id=f"subtask_2_{i+1}",
            cot_agent_desc=cot_agent_desc1
        )
        agents.append(f"CoT agent {results1['cot_agent'].id}, converting equation {i+1}, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
        sub_tasks.append(f"Sub-task 2.{i+1} output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
        logs.append(results1['subtask_desc'])
        candidate_relationships.append(results1['answer'].content)

    aggregate_instruction2 = "Subtask 3: Merge the derived relationships from both equations into a unified system"
    aggregate_desc2 = {
        'instruction': aggregate_instruction2,
        'input': [taskInfo] + candidate_relationships,
        'temperature': 0.0,
        'context': ["user query", "derived relationships from equations"]
    }
    results2 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc2
    )
    agents.append(f"Aggregate agent {results2['aggregate_agent'].id}, consolidating relationships, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    review_instruction6 = "Subtask 7: Refine the step-by-step solution explanation for clarity and coherence"
    review_desc6 = {
        'instruction': review_instruction6,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "consolidated relationships"]
    }
    results6 = await self.review(
        subtask_id="subtask_7",
        review_desc=review_desc6
    )
    agents.append(f"Review agent {results6['review_agent'].id}, refining clarity, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    specific_format_instruction7 = "Subtask 8: Format the final result and answer (xy = 25) to meet output requirements"
    specific_format_desc7 = {
        'instruction': specific_format_instruction7,
        'input': [taskInfo, results6['answer'].content],
        'temperature': 0.0,
        'context': ["user query", "refined solution"],
        'format': 'short and concise, without explanation'
    }
    results7 = await self.specific_format(
        subtask_id="subtask_8",
        formatter_desc=specific_format_desc7
    )
    agents.append(f"SpecificFormat agent {results7['formatter_agent'].id}, formatting final answer, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    conditional_check = 'x and y greater than 1' in results2['answer'].content or 'domain' in results2['answer'].content
    if conditional_check:
        specific_format_instruction4 = "Subtask 5: Identify domain constraints (x,y>1), clarify assumptions, and validate consistency of derived equations"
        specific_format_desc4 = {
            'instruction': specific_format_instruction4,
            'input': [taskInfo, results6['answer'].content, results7['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "refined solution", "formatted answer"],
            'format': 'short and concise, without explanation'
        }
        results4 = await self.specific_format(
            subtask_id="subtask_4",
            formatter_desc=specific_format_desc4
        )
        agents.append(f"SpecificFormat agent {results4['formatter_agent'].id}, validating domain constraints, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])

        answer_generate_instruction3 = "Subtask 6: Check for missing algebraic steps or domain considerations (x,y>1) and fill in any gaps"
        answer_generate_desc3 = {
            'instruction': answer_generate_instruction3,
            'input': [taskInfo, results4['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "domain validation"]
        }
        results3 = await self.answer_generate(
            subtask_id="subtask_6",
            cot_agent_desc=answer_generate_desc3
        )
        agents.append(f"AnswerGenerate agent {results3['cot_agent'].id}, filling gaps, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Sub-task 6 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])

        review_instruction5 = "Subtask 7: Validate each algebraic manipulation step leading to 100 = 4xy"
        review_desc5 = {
            'instruction': review_instruction5,
            'input': [taskInfo, results3['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "gap filled solution"]
        }
        results5 = await self.review(
            subtask_id="subtask_5",
            review_desc=review_desc5
        )
        agents.append(f"Review agent {results5['review_agent'].id}, validating algebraic steps, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
        sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
        logs.append(results5['subtask_desc'])

        specific_format_instruction5 = "Subtask 9: Format the validated solution to meet output requirements"
        specific_format_desc5 = {
            'instruction': specific_format_instruction5,
            'input': [taskInfo, results5['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "validated solution"],
            'format': 'short and concise, without explanation'
        }
        results8 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=specific_format_desc5
        )
        agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, formatting validated solution, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Sub-task 9 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
        return final_answer, logs
    else:
        specific_format_instruction4_false = "Subtask 5: Identify domain constraints (x,y>1), clarify assumptions, and validate consistency of derived equations"
        specific_format_desc4_false = {
            'instruction': specific_format_instruction4_false,
            'input': [taskInfo, results6['answer'].content, results7['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "refined solution", "formatted answer"],
            'format': 'short and concise, without explanation'
        }
        results4_false = await self.specific_format(
            subtask_id="subtask_4_false",
            formatter_desc=specific_format_desc4_false
        )
        agents.append(f"SpecificFormat agent {results4_false['formatter_agent'].id}, validating domain constraints (false branch), thinking: {results4_false['thinking'].content}; answer: {results4_false['answer'].content}")
        sub_tasks.append(f"Sub-task 4 (false branch) output: thinking - {results4_false['thinking'].content}; answer - {results4_false['answer'].content}")
        logs.append(results4_false['subtask_desc'])

        answer_generate_instruction3_false = "Subtask 6: Check for missing algebraic steps or domain considerations (x,y>1) and fill in any gaps"
        answer_generate_desc3_false = {
            'instruction': answer_generate_instruction3_false,
            'input': [taskInfo, results4_false['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "domain validation (false branch)"]
        }
        results3_false = await self.answer_generate(
            subtask_id="subtask_6_false",
            cot_agent_desc=answer_generate_desc3_false
        )
        agents.append(f"AnswerGenerate agent {results3_false['cot_agent'].id}, filling gaps (false branch), thinking: {results3_false['thinking'].content}; answer: {results3_false['answer'].content}")
        sub_tasks.append(f"Sub-task 6 (false branch) output: thinking - {results3_false['thinking'].content}; answer - {results3_false['answer'].content}")
        logs.append(results3_false['subtask_desc'])

        review_instruction5_false = "Subtask 7: Validate each algebraic manipulation step leading to 100 = 4xy"
        review_desc5_false = {
            'instruction': review_instruction5_false,
            'input': [taskInfo, results3_false['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "gap filled solution (false branch)"]
        }
        results5_false = await self.review(
            subtask_id="subtask_5_false",
            review_desc=review_desc5_false
        )
        agents.append(f"Review agent {results5_false['review_agent'].id}, validating algebraic steps (false branch), thinking: {results5_false['thinking'].content}; answer: {results5_false['answer'].content}")
        sub_tasks.append(f"Sub-task 5 (false branch) output: thinking - {results5_false['thinking'].content}; answer - {results5_false['answer'].content}")
        logs.append(results5_false['subtask_desc'])

        specific_format_instruction5_false = "Subtask 9: Format the validated solution to meet output requirements"
        specific_format_desc5_false = {
            'instruction': specific_format_instruction5_false,
            'input': [taskInfo, results5_false['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "validated solution (false branch)"],
            'format': 'short and concise, without explanation'
        }
        results8_false = await self.specific_format(
            subtask_id="subtask_9_false",
            formatter_desc=specific_format_desc5_false
        )
        agents.append(f"SpecificFormat agent {results8_false['formatter_agent'].id}, formatting validated solution (false branch), thinking: {results8_false['thinking'].content}; answer: {results8_false['answer'].content}")
        sub_tasks.append(f"Sub-task 9 (false branch) output: thinking - {results8_false['thinking'].content}; answer - {results8_false['answer'].content}")
        logs.append(results8_false['subtask_desc'])

        final_answer_false = await self.make_final_answer(results8_false['thinking'], results8_false['answer'], sub_tasks, agents)
        return final_answer_false, logs

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
