async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction2 = "Subtask 2: Extract and list the essential features: triangle ABC with sides 5, 9, 10; circle ω; tangents at B, C; intersection D; line AD meets ω again at P"
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting essential features, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    cot_sc_instruction1 = "Subtask 1: Evaluate and confirm the properties of inputs: verify side lengths, define circumcircle and tangent properties, with context from extracted features"
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results1['list_thinking']):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, confirming properties, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_sc_instruction0 = "Subtask 0: Categorize the problem type and identify suitable techniques (power of a point, symmedian) based on confirmed properties"
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results0 = await self.sc_cot(
        subtask_id="subtask_0",
        cot_sc_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results0['list_thinking']):
        agents.append(f"CoT-SC agent {results0['cot_agent'][idx].id}, categorizing problem, thinking: {results0['list_thinking'][idx]}; answer: {results0['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 0 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])
    
    candidate_solutions = []
    for i in range(2):
        answer_generate_instruction3 = "Subtask 3: Generate initial candidate solution approaches (e.g., power-of-a-point vs. symmedian method)"
        answer_generate_desc3 = {
            'instruction': answer_generate_instruction3,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results3 = await self.answer_generate(
            subtask_id="subtask_3",
            cot_agent_desc=answer_generate_desc3
        )
        agents.append(f"AnswerGenerate agent {results3['cot_agent'].id}, generating candidate solution approach, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        candidate_solutions.append((results3['thinking'], results3['answer']))
    
    aggregate_instruction4 = "Subtask 4: Aggregate candidate approaches, validate consistency, and select the most coherent method"
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo] + [item[1] for item in candidate_solutions],
        'temperature': 0.0,
        'context': ["user query", "candidate solutions"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating candidate solutions, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    review_instruction5 = "Subtask 5: Enhance clarity and coherence of the selected solution outline"
    review_desc5 = {
        'instruction': review_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc5
    )
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing solution clarity, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    formatter_instruction6 = "Subtask 6: Format the finalized solution into the required JSON structure"
    formatter_desc6 = {
        'instruction': formatter_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "review feedback"],
        'format': 'short and concise, without explanation'
    }
    results6 = await self.specific_format(
        subtask_id="subtask_6",
        formatter_desc=formatter_desc6
    )
    agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, formatting final solution, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    condition = 'primary method success' in results4['answer'].content.lower()
    if condition:
        
        final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
        return final_answer, logs
    else:
        
        formatter_instruction7 = "Subtask 7: Identify, clarify, and validate units or elements in the alternative approach if the primary method fails"
        formatter_desc7 = {
            'instruction': formatter_instruction7,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "failed primary method"]
        }
        results7 = await self.specific_format(
            subtask_id="subtask_7",
            formatter_desc=formatter_desc7
        )
        agents.append(f"SpecificFormat agent {results7['formatter_agent'].id}, clarifying alternative approach, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])
        
        answer_generate_instruction8 = "Subtask 8: Validate and assess the alternative approach’s intermediate results"
        answer_generate_desc8 = {
            'instruction': answer_generate_instruction8,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "alternative approach"]
        }
        results8 = await self.review(
            subtask_id="subtask_8",
            review_desc=answer_generate_desc8
        )
        agents.append(f"Review agent {results8['review_agent'].id}, validating alternative approach, feedback: {results8['thinking'].content}; correct: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: feedback - {results8['thinking'].content}; correct - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])
        
        formatter_instruction9 = "Subtask 9: Enhance clarity and coherence of the alternative approach articulation"
        formatter_desc9 = {
            'instruction': formatter_instruction9,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'temperature': 0.0,
            'context': ["user query", "alternative approach review"],
            'format': 'short and concise, without explanation'
        }
        results9 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=formatter_desc9
        )
        agents.append(f"SpecificFormat agent {results9['formatter_agent'].id}, formatting alternative solution, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])
        
        formatter_instruction6_alt = "Subtask 6: Format the finalized alternative solution into the required JSON structure"
        formatter_desc6_alt = {
            'instruction': formatter_instruction6_alt,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "alternative solution"],
            'format': 'short and concise, without explanation'
        }
        results6_alt = await self.specific_format(
            subtask_id="subtask_6_alt",
            formatter_desc=formatter_desc6_alt
        )
        agents.append(f"SpecificFormat agent {results6_alt['formatter_agent'].id}, formatting final alternative solution, thinking: {results6_alt['thinking'].content}; answer: {results6_alt['answer'].content}")
        sub_tasks.append(f"Subtask 6_alt output: thinking - {results6_alt['thinking'].content}; answer - {results6_alt['answer'].content}")
        logs.append(results6_alt['subtask_desc'])
        
        final_answer = await self.make_final_answer(results6_alt['thinking'], results6_alt['answer'], sub_tasks, agents)
        return final_answer, logs
