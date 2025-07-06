async def forward_10(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Calculate the total cost of the jacket and two pairs of shoes, given jacket costs $30 and each pair of shoes costs $20."
    cot_agent_desc_1 = {
        'instruction': cot_instruction_1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc_1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, calculating total cost, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = "Subtask 2: Calculate Sara's total earnings from babysitting 4 times at $5 each."
    cot_agent_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc_2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, calculating babysitting earnings, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction_3 = "Subtask 3: Calculate Sara's total money after babysitting and adding initial savings of $10, using babysitting earnings from Subtask 2."
    cot_agent_desc_3 = {
        'instruction': cot_instruction_3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc_3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, calculating total money after babysitting and savings, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    mow_count = 0
    total_money = None
    total_cost = None

    try:
        total_cost = float(results1['answer'].content.strip().replace('$',''))
    except:
        total_cost = None
    try:
        total_money = float(results3['answer'].content.strip().replace('$',''))
    except:
        total_money = None

    while total_money is not None and total_cost is not None and total_money < total_cost:
        mow_count += 1
        cot_instruction_4 = f"Subtask 4: Add $4 to Sara's total money for mowing the lawn {mow_count} time(s), starting from previous total money ${total_money}."
        cot_agent_desc_4 = {
            'instruction': cot_instruction_4,
            'input': [taskInfo, str(total_money), str(mow_count)],
            'temperature': 0.0,
            'context': ["user query", "previous total money", "mow count"]
        }
        results4 = await self.cot(
            subtask_id=f"subtask_4_{mow_count}",
            cot_agent_desc=cot_agent_desc_4
        )
        agents.append(f"CoT agent {results4['cot_agent'].id}, adding mowing earnings, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Subtask 4_{mow_count} output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])

        try:
            total_money = float(results4['answer'].content.strip().replace('$',''))
        except:
            break

        cot_instruction_5 = f"Subtask 5: Check if Sara's total money ${total_money} is enough to cover total cost ${total_cost}."
        cot_agent_desc_5 = {
            'instruction': cot_instruction_5,
            'input': [taskInfo, str(total_money), str(total_cost)],
            'temperature': 0.0,
            'context': ["user query", "total money", "total cost"]
        }
        results5 = await self.cot(
            subtask_id=f"subtask_5_{mow_count}",
            cot_agent_desc=cot_agent_desc_5
        )
        agents.append(f"CoT agent {results5['cot_agent'].id}, checking affordability, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
        sub_tasks.append(f"Subtask 5_{mow_count} output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
        logs.append(results5['subtask_desc'])

        if 'enough' in results5['answer'].content.lower() or total_money >= total_cost:
            break

    aggregate_instruction_6 = f"Subtask 6: Consolidate the final count of times Sara must mow the lawn ({mow_count}) to afford the jacket and shoes costing ${total_cost}."
    aggregate_desc_6 = {
        'instruction': aggregate_instruction_6,
        'input': [taskInfo, str(mow_count), str(total_cost)],
        'temperature': 0.0,
        'context': ["user query", "mow count", "total cost"]
    }
    results6 = await self.aggregate(
        subtask_id="subtask_6",
        aggregate_desc=aggregate_desc_6
    )
    agents.append(f"Aggregate agent {results6['aggregate_agent'].id}, consolidating mow count, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    programmer_instruction_7 = f"Subtask 7: Validate the correctness of the calculated number of times Sara must mow the lawn ({mow_count}) to afford the jacket and shoes costing ${total_cost}, using previous calculations."
    programmer_desc_7 = {
        'instruction': programmer_instruction_7,
        'input': [taskInfo, results6['thinking'], results6['answer'], results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregate result", "total cost", "total money"]
    }
    results7 = await self.programmer(
        subtask_id="subtask_7",
        programmer_desc=programmer_desc_7
    )
    agents.append(f"Programmer agent {results7['programmer_agent'].id}, validating mow count, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}; output: {results7['exec_result']}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}; output - {results7['exec_result']}")
    logs.append(results7['subtask_desc'])

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
