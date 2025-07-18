INTERACTION_PATTERN = """
Sample agent iteraction pattern:
Chain-of-Thought: 
```python
    # Sub-task 1: Analyze first expression/data component with self-consistency
    cot_instruction1 = "Sub-task 1: Consider/calculate all possible cases of [problem #1], with context ...."
    cot_agent_desc = {{
        'instruction': cot_instruction1, 
        'input': [taskInfo], 
        'temperature': 0.0, 
        'context_desc': ["user query", ....]
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
    final_decision_instruction2 = "Sub-task 2: Synthesize and choose the most consistent answer for [problem]." # must contain if use this pattern
    N = self.max_sc
    
    cot_sc_desc2 = {{
        'instruction': cot_sc_instruction2, 
        'final_decision_instruction': final_decision_instruction2,  # must contain if use this pattern
        'input': [taskInfo, thinking1, answer1], 
        'temperature': 0.5, 
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1"]
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
    cot_reflect_instruction3 = "Sub-task 3: Your problem is ... [problem]."
    critic_instruction3 = ""Please review and provide the limitations of provided solutions of ....."  # must contain if use this pattern
    cot_reflect_desc3 = {{
        'instruction': cot_reflect_instruction3, 
        'critic_instruction': critic_instruction3,  # must contain if use this pattern
        'input': [taskInfo, thinking1, answer1, thinking2, answer2], 
        'temperature': 0.0, 
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }}
    
    results3, log3 = await self.reflexion(
        subtask_id="subtask_3", 
        reflect_desc=cot_reflect_desc3, 
        n_repeat=self.max_round
    )
    
    logs.append(log3)
```

Debate:
```python
    debate_instruction5 = "Sub-task 5: Your problem is .... [instruction]."
    final_decision_instruction5 = "Sub-task 5: [problem]" # must contain if use this pattern

    debate_desc5 = {{
        "instruction": debate_instruction5,
        "final_decision_instruction": final_decision_instruction5,  # must contain if use this pattern
        "input": [taskInfo, thinking4, answer4],
        "context_desc": ["user query", "thinking of subtask 4", "answer of subtask 4"],
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
    logs = []

    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage 1: Expression Analysis]
    [Objective] 
    - Analyze the mathematical or logical expression in the given task.
    - Extract behavior, range, and key characteristics using CoT reasoning.
    """
    # --------------------------------------------------------------------------------------------------------------

    # Sub-task 1: Analyze first expression/data component with self-consistency
    cot_instruction1 = (
        "Sub-task 1: Analyze [expression #1], determining its behavior, range, "
        "and key characteristics with context from [taskInfo]"
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"],
    }
    results1, log1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
    )
    logs.append(log1)

    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage 2: Comparative Reasoning]
    [Objective] 
    - Compare results from multiple expressions or subtasks.
    - Synthesize insights across components.
    """
    # --------------------------------------------------------------------------------------------------------------

    """
    <This section is just to describe content in one stage. Do not include them in generated code>
    [Stage 3: Final Synthesis]
    [Objective] 
    - Synthesize all subtasks into a coherent final answer.
    - Apply reasoning to derive a conclusive insight.
    """
    # --------------------------------------------------------------------------------------------------------------

    final_answer = await self.make_final_answer(resultsn['thinking'], resultsn['answer'])
    return final_answer, logs
'''