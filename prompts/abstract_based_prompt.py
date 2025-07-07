INTERACTION_PATTERN = """
Sample agent iteraction pattern:
Chain-of-Thought: 
```python
    # Sub-task 1: Analyze first expression/data component with self-consistency
    cot_instruction1 = "Sub-task 1: Analyze [expression #1], determining its behavior, range, and key characteristics with context from [taskInfo]"
    cot_agent_desc = {{
        'instruction': cot_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query", ....]
    }}
    results1 = await self.cot(
        subtask_id="subtask_1", 
        cot_agent_desc=cot_agent_desc
    )
    
    agents.append(f"CoT agent {{results1['cot_agent'].id}}, analyzing [expression #1], thinking: {{results1['thinking'].content}}; answer: {{results1['answer'].content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{results1['thinking'].content}}; answer - {{results1['answer'].content}}")
    logs.append(results1['subtask_desc'])
```

Self-Consistency Chain-of-Thought:
```python
    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, consider/calculate potential cases of [problem #2], with context ....."
    N = self.max_sc
    
    cot_sc_desc = {{
        'instruction': cot_sc_instruction2, 
        'input': [taskInfo, thinking1, answer1], 
        'temperature': 0.5, 
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }}
    
    results2 = await self.sc_cot(
        subtask_id="subtask_2", 
        cot_sc_desc=cot_sc_desc, 
        n_repeat=self.max_sc
    )
    
    sub_tasks.append(f"Sub-task 2 output: thinking - {{results2['thinking'].content}}; answer - {{results2['answer'].content}}")
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {{results2['cot_agent'][idx].id}}, consider all possible cases of [problem #2], thinking: {{results2['list_thinking'][idx]}}; answer: {{results2['list_answer'][idx]}}")
    logs.append(results2['subtask_desc'])
```

Reflexion:
```python
    cot_reflect_instruction3 = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, filter the valid scenarios that meet the [conditions stated in the queries]."
    critic_instruction3 = "Please review the [valid scenarios] filtering and provide its limitations."
    cot_reflect_desc3 = {{
        'instruction': cot_reflect_instruction3, 'input': [taskInfo, thinking1, answer1, thinking2, answer2], 'output': ["thinking", "answer"], 
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }}
    critic_desc3 = {{
        'instruction': critic_instruction3, 'output': ["feedback", "correct"], 'temperature': 0.0
    }}
    
    results3 = await  self.reflexion(
        subtask_id="subtask_3", 
        cot_reflect_desc=cot_reflect_desc3, 
        critic_desc=critic_desc3, 
        n_repeat=self.max_round
    )
    
    agents.append(f"Reflexion CoT agent {{results3['cot_agent'].id}}, filter valid scenarios of [problem], thinking: {{results3['list_thinking'][0].content}}; answer: {{results3['list_answer'][0].content}}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {{results3['critic_agent'].id}}, providing feedback, thinking: {{results3['list_feedback'][i].content}}; answer: {{results3['list_correct'][i].content}}")
        if k + 1 < len(results5['list_thinking']) and k + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining final answer, thinking: {results3['list_thinking'][k + 1].content}; answer: {results3['list_answer'][k + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {{results3['thinking'].content}}; answer - {{results3['answer'].content}}")
    logs.append(results3['subtask_desc'])
```

Debate:
```python
    debate_instruction_5 = "Sub-task 5: Based on the output of Sub-task 4, convert [intermediate output] into [specific format] and calculate [the final answer]"
    final_decision_instruction_5 = "Sub-task 5: Make final decision on [final output]."
    
    debate_desc5 = {{
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "input": [taskInfo, thinking4, answer4],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }}
    
    final_decision_desc5 = {{
        "instruction": final_decision_instruction_5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }}
    
    results5 = await self.debate(
        subtask_id="subtask_5", 
        debate_desc=debate_desc5, 
        final_decision_desc=final_decision_desc5, 
        n_repeat=self.max_round
    ):
    
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {{agent.id}}, round {{round}}, converting [intermediate output] and calculating [final output], thinking: {{results5['list_thinking'][round][idx].content}}; answer: {{results5['list_answer'][round][idx].content}}")

    agents.append(f"Final Decision agent, calculating [final output], thinking: {{results5['thinking'].content}}; answer: {{results5['answer'].content}}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {{results5['thinking'].content}}; answer - {{results5['answer'].content}}")
    logs.append(results5['subtask_desc'])
```
AnswerGenerate
```python
    cot_agent_instruction1 = "Sub-task 1: Analyze [expression #1], determining its behavior, range, and key characteristics with context from [taskInfo]"
    cot_agent_desc = {{
        'instruction': cot_agent_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query", ....]
    }}
    results1 = await self.answer_generate(
        subtask_id="subtask_1", 
        cot_agent_desc=cot_agent_desc
    )

    agents.append(f"CoT agent {{results1['cot_agent'].id}}, analyzing [expression #1], thinking: {{results1['thinking'].content}}; answer: {{results1['answer'].content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{results1['thinking'].content}}; answer - {{results1['answer'].content}}")
    logs.append(results1['subtask_desc'])
```

SpecificFormat
```python
    formatter_instruction1 = "Sub-task 1: Analyze [expression #1], determining its behavior, range, and key characteristics with context from [taskInfo]"
    formatter_desc = {{
        'instruction': formatter_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query", ....],
        'format': 'short and concise, without explaination'
    }}
    results1 = await self.specific_format(
        subtask_id="subtask_1", 
        formatter_desc=formatter_desc
    )

    agents.append(f"CoT agent {{results1['formatter_agent'].id}}, analyzing [expression #1], thinking: {{results1['thinking'].content}}; answer: {{results1['answer'].content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{results1['thinking'].content}}; answer - {{results1['answer'].content}}")
    logs.append(results1['subtask_desc'])
```

AggregateAgent
```python
    aggregate_instruction2 = "Sub-task 2: From solutions generated in Subtask 1, aggregate these solutions and return the consistent and the best solution for [subtask]"
    aggregate_desc = {{
        'instruction': aggregate_instruction2, 
        'input': [taskInfo] + [(previous solutions that need to be aggregated)], 
        'temperature': 0.0, 
        'context': ["user query", "solutions generated from subtask 1"]
    }}
    results2 = await self.aggregate(
        subtask_id="subtask_1", 
        aggregate_desc=aggregate_desc
    )
    
    agents.append(f"CoT agent {{results2['aggregate_agent'].id}}, analyzing [expression #1], thinking: {{results2['thinking'].content}}; answer: {{results2['answer'].content}}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {{results2['thinking'].content}}; answer - {{results2['answer'].content}}")
    logs.append(results2['subtask_desc'])
```

CodeGenerate
```python
    code_generate_instruction1 = "Sub-task 1: Generate Python runnable code that addresses the following problem: [problem1]"
    code_generate_desc1 = {{
        'instruction': code_generate_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query", ....],
        'entry_point': "entry_point that is suitable for this subtask"
    }}
    results1 = await self.code_generate(
        subtask_id="subtask_1", 
        code_generate_desc=code_generate_desc1
    )
    
    agents.append(f"Code Generate Agent {{results1['code_generate_agent'].id}}, generate code for problem [problem #1], thinking: {{results1['thinking'].content}}; code: {{results1['answer'].content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{results1['thinking'].content}}; code - {{results1['answer'].content}}")
    logs.append(results1['subtask_desc'])
```

ProgrammerAgent
```python
    programmer_instruction1 = "Sub-task 1: Generate Python runnable code that addresses the following problem: [problem1]"
    programmer_desc1 = {{
        'instruction': programmer_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query", ....],
        'entry_point': "entry_point that is suitable for this subtask"
    }}
    results1 = await self.programmer(
        subtask_id="subtask_1", 
        programmer_desc=programmer_desc1
    )
    
    agents.append(f"Programmer Agent {{results1['programmer_agent'].id}}, generate code for problem [problem #1], thinking: {{results1['thinking'].content}}; answer: {{results1['answer'].content}}, executing reults: {{results1['exec_result']}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{results1['thinking'].content}}; answer - {{results1['answer'].content}}; output - {{results1['exec_result']}}")
    logs.append(results1['subtask_desc'])
```

ReviseAgent
```python
    revise_instruction2 = "Subtask 2: Revise previous solutions of problem: [problem 1]"
    revise_desc2 = {{
        'instruction': revise_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }}
    
    results2 = await self.revise(
        subtask_id = "subtask_2", 
        revise_desc = revise_desc2
    )
    
    agents.append(f"Revise agent {{results2['revise_agent'].id}}, revise solution from subtask 1, feedback: [feedback], thinking: {{results2['thinking'].content}}; revised_solution: {{results2['revised_solution'].content}}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {{results2['thinking'].content}}; revised_solution - {{results2['revised_solution'].content}}")
    logs.append(results2['subtask_desc'])
```

ReviewAgent
```python
    review_instruction2 = "Subtask 2: Review previous solutions of problem: [problem 1]"
    review_desc2 = {{
        'instruction': review_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 1', 'answer of subtask 1']
    }}
    
    results2 = await self.review(
        subtask_id = "subtask_2", 
        review_desc = review_desc2
    )
    
    agents.append(f"Review agent {{results2['review_agent'].id}}, review solution from subtask 1, feedback: [feedback], feedback: {{results2['thinking'].content}}; correct: {{results2['answer'].content}}")
    sub_tasks.append(f"Sub-task 2 output: feedback - {{results2['thinking'].content}}; correct - {{results2['answer'].content}}")
    logs.append(results2['subtask_desc'])
```
"""