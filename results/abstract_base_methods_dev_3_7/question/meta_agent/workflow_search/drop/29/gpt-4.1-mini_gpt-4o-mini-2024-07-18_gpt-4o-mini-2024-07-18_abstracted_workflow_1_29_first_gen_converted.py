async def forward_29(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Extract the total population number from the passage to use as the base for percentage calculations."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting total population, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 2: Extract the percentage of people of German ancestry from the passage."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting German ancestry percentage, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = "Subtask 3: Extract the percentages of people from US and Danish ancestry from the passage."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, extracting US and Danish ancestry percentages, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_instruction4 = "Subtask 4: Calculate the number of people of German ancestry by applying the German percentage to the total population."
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, calculating German ancestry count, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_instruction5 = "Subtask 5: Calculate the number of people of US ancestry by applying the US percentage to the total population."
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"]
    }
    results5 = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5
    )
    agents.append(f"CoT agent {results5['cot_agent'].id}, calculating US ancestry count, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_instruction6 = "Subtask 6: Calculate the number of people of Danish ancestry by applying the Danish percentage to the total population."
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, calculating Danish ancestry count, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_instruction7 = "Subtask 7: Sum the number of people from US and Danish ancestries to get the combined total."
    cot_agent_desc7 = {
        'instruction': cot_instruction7,
        'input': [results5['thinking'], results5['answer'], results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7 = await self.cot(
        subtask_id="subtask_7",
        cot_agent_desc=cot_agent_desc7
    )
    agents.append(f"CoT agent {results7['cot_agent'].id}, summing US and Danish ancestry counts, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_instruction8 = "Subtask 8: Calculate how many more people were German than the combined US and Danish ancestries by subtracting the combined total from the German total."
    cot_agent_desc8 = {
        'instruction': cot_instruction8,
        'input': [results4['thinking'], results4['answer'], results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 7", "answer of subtask 7"]
    }
    results8 = await self.cot(
        subtask_id="subtask_8",
        cot_agent_desc=cot_agent_desc8
    )
    agents.append(f"CoT agent {results8['cot_agent'].id}, calculating difference between German and combined US/Danish counts, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
    return final_answer, logs
