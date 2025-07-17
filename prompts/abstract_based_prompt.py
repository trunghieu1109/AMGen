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
    results1, log1 = await self.cot(
        subtask_id="subtask_1", 
        cot_agent_desc=cot_agent_desc
    )
    
    logs.append(log1)
```

Self-Consistency Chain-of-Thought:
```python
    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, consider/calculate potential cases of [problem #2], with context ....."
    N = self.max_sc
    
    cot_sc_desc2 = {{
        'instruction': cot_sc_instruction2, 
        'input': [taskInfo, thinking1, answer1], 
        'temperature': 0.5, 
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }}
    
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2", 
        cot_agent_desc=cot_sc_desc2, 
        n_repeat=self.max_sc
    )
    
    logs.append(log2)
```

Reflexion:
```python
    cot_reflect_instruction3 = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, filter the valid scenarios that meet the [conditions stated in the queries]."
    critic_instruction3 = "Please review the [valid scenarios] filtering and provide its limitations."
    cot_reflect_desc3 = {{
        'instruction': cot_reflect_instruction3, 'input': [taskInfo, thinking1, answer1, thinking2, answer2], 'output': ["thinking", "answer"], 
        'temperature': 0.0, 'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }}
    
    results3, log3 = await  self.reflexion(
        subtask_id="subtask_3", 
        reflect_desc=cot_reflect_desc3, 
        n_repeat=self.max_round
    )
    
    logs.append(log3)
```

Debate:
```python
    debate_instruction_5 = "Sub-task 5: Based on the output of Sub-task 4, convert [intermediate output] into [specific format] and calculate [the final answer]"

    debate_desc5 = {{
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "input": [taskInfo, thinking4, answer4],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }}
    
    results5, log5 = await self.debate(
        subtask_id="subtask_5", 
        debate_desc=debate_desc5, 
        n_repeat=self.max_round
    ):
    
    logs.append(log5)
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
    results1, log1 = await self.answer_generate(
        subtask_id="subtask_1", 
        cot_agent_desc=cot_agent_desc
    )

    logs.append(log1)
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
    results1, log1 = await self.specific_format(
        subtask_id="subtask_1", 
        formatter_desc=formatter_desc
    )

    logs.append(log1)
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
    results2, log2 = await self.aggregate(
        subtask_id="subtask_1", 
        aggregate_desc=aggregate_desc
    )
    
    logs.append(log2)
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
    results1, log1 = await self.code_generate(
        subtask_id="subtask_1", 
        code_generate_desc=code_generate_desc1
    )
    
    logs.append(log1)
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
    results1, log1 = await self.programmer(
        subtask_id="subtask_1", 
        programmer_desc=programmer_desc1
    )
    
    logs.append(log1)
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
    
    results2, log2 = await self.revise(
        subtask_id = "subtask_2", 
        revise_desc = revise_desc2
    )
    
    logs.append(log2)
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
    
    results2, log2 = await self.review(
        subtask_id = "subtask_2", 
        review_desc = review_desc2
    )
    
    logs.append(log2)
```
"""

ABSTRACTED_WORKFLOW_TEMPLATE = '''
async def forward(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    # Initialize lists to keep track of sub-tasks and agents
    logs =  []
    
    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage 1: <Fill the stage 1's stage_name>]
    [Objective] 
    - <Describe in detail the abstracted objective of stage 1.>
    - <Describe in detail the abstracted objective of stage 1.>
    """
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    # Sub-task 1: Analyze first expression/data component with self-consistency
    cot_instruction1 = "Sub-task 1: Analyze [expression #1], determining its behavior, range, and key characteristics with context from [taskInfo]"
    cot_agent_desc = {{
        'instruction': cot_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context': ["user query"]
    }}
    results1, log1 = await self.cot(
        subtask_id="subtask_1", 
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    <Continue with next stages>
    
    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage n: <Fill the stage n's stage_name>]
    [Objective] 
    - <Describe in detail the abstracted objective of stage n.>
    - <Describe in detail the abstracted objective of stage n.>
    """
    
    # --------------------------------------------------------------------------------------------------------------
    
    <At this section, you implement only one subtask that could appear in this stage, by applying one agent collaboration patterns>
    
    final_answer = await self.make_final_answer(resultsn['thinking'], resultsn['answer'])
    return final_answer, logs
'''