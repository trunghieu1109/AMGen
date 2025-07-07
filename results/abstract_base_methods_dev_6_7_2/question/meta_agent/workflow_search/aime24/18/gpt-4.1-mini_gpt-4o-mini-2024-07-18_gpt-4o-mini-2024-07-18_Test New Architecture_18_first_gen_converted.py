async def forward_18(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Formulate equations for points P on the x-axis and Q on the y-axis so that segment PQ has length 1 and derive any constraints or loci." 
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing segment PQ constraints, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Based on the output from Subtask 1, consider and calculate potential cases for the locus of point C on segment AB." 
    N = self.max_sc
    cot_sc_desc = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc,
        n_repeat=N
    )
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, considering candidate loci for C, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    logs.append(results2['subtask_desc'])

    candidates = []
    for i in range(self.max_sc):
        candidates.append((results2['list_thinking'][i], results2['list_answer'][i]))

    aggregate_instruction3 = "Subtask 3: Aggregate multiple candidate locus descriptions to produce a coherent set of possible C positions." 
    aggregate_desc = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo] + candidates,
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 2"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, consolidating candidate loci, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_instruction4 = "Subtask 4: Systematically generate potential coordinates for C from the aggregated locus data." 
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, generating candidate coordinates for C, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_instruction5 = "Subtask 5: Detect any missing or inconsistent cases in the enumeration of candidate C points and supply the missing analyses." 
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.answer_generate(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5
    )
    agents.append(f"AnswerGenerate agent {results5['cot_agent'].id}, identifying and filling gaps, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    condition_valid_unique = 'valid and unique candidate C' in results5['answer'].content.lower()

    if condition_valid_unique:
        pass
    else:
        formatter_instruction6 = "Subtask 6: Identify and clarify the coordinate units of the invalid candidate, validate each against problem criteria, and isolate the truly unique C." 
        formatter_desc6 = {
            'instruction': formatter_instruction6,
            'input': [taskInfo, results5['thinking'], results5['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 5", "answer of subtask 5"],
            'format': 'short and concise, without explanation'
        }
        results6 = await self.specific_format(
            subtask_id="subtask_6",
            formatter_desc=formatter_desc6
        )
        agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, clarifying and validating units, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])

    review_instruction7 = "Subtask 7: Evaluate the derived expression for OC^2, confirm it is simplified p/q with p,q relatively prime." 
    review_desc7 = {
        'instruction': review_instruction7,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results7 = await self.review(
        subtask_id="subtask_7",
        review_desc=review_desc7
    )
    agents.append(f"Review agent {results7['review_agent'].id}, validating OC^2 expression, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    formatter_instruction8 = "Subtask 8: Refine and restructure the solution explanation for maximum clarity and logical coherence." 
    formatter_desc8 = {
        'instruction': formatter_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "feedback of subtask 7", "correct of subtask 7"]
    }
    results8 = await self.specific_format(
        subtask_id="subtask_8",
        formatter_desc=formatter_desc8
    )
    agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, enhancing clarity and coherence, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    answer_generate_instruction9 = "Subtask 9: Format the final numeric answer p+q according to contest output rules." 
    answer_generate_desc9 = {
        'instruction': answer_generate_instruction9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
    }
    results9 = await self.answer_generate(
        subtask_id="subtask_9",
        cot_agent_desc=answer_generate_desc9
    )
    agents.append(f"AnswerGenerate agent {results9['cot_agent'].id}, formatting final answer, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
