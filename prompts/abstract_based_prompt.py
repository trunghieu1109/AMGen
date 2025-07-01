INTERACTION_PATTERN = """
Sample agent iteraction pattern:
Chain-of-Thought: 
```python
    cot_instruction = "Sub-task 1: Consider/calculate all possible cases of [problem #1], with context ...."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc1 = {{
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {{cot_agent.id}}, consider/calculate all possible scenarios of [problem #1], thinking: {{thinking1.content}}; answer: {{answer1.content}}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {{thinking1.content}}; answer - {{answer1.content}}")
    subtask_desc1['response'] = {{
        "thinking": thinking1,
        "answer": answer1
    }}
    logs.append(subtask_desc1)
```

Self-Consistency Chain-of-Thought:
```python
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, consider/calculate potential cases of [problem #2], with context ....."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {{}}
    answermapping = {{}}
    subtask_desc2 = {{
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }}
    for i in range(N):
        # Each CoT-SC agent tries to calculate all possible cases independently
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {{cot_agents[i].id}}, consider all possible cases of [problem #2], thinking: {{thinking2.content}}; answer: {{answer2.content}}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
        
    # The most common answer is chosen for consistency and accuracy.
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {{thinking2.content}}; answer - {{answer2.content}}")
    subtask_desc2['response'] = {{
        "thinking": thinking2,
        "answer": answer2
    }}
    logs.append(subtask_desc2)
```

Reflexion:
```python
    cot_reflect_instruction = "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, filter the valid scenarios that meet the [conditions stated in the queries]."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    
    # Input for CoT agent
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    
    subtask_desc3 = {{
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }}
    
    # Generate the first version
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {{cot_agent.id}}, filter valid scenarios of [problem], thinking: {{thinking3.content}}; answer: {{answer3.content}}")

    for i in range(N_max):
        # Critic agent debates and criticizes pros and cons of previous version
        feedback, correct = await critic_agent(cot_inputs + [thinking3, answer3], 
                                       "please review the [valid scenarios] filtering and provide its limitations.", 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {{critic_agent.id}}, providing feedback, thinking: {{feedback.content}}; answer: {{correct.content}}")
        if correct.content == "True":
            break
        
        # Include previous version and feedback from critic agent as input
        cot_inputs.extend([thinking3, answer3, feedback])
        
        # Generate new version based on previous version and feedback
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {{cot_agent.id}}, refining valid scenarios of [problem], thinking: {{thinking3.content}}; answer: {{answer3.content}}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {{thinking3.content}}; answer - {{answer3.content}}")
    subtask_desc3['response'] = {{
        "thinking": thinking3,
        "answer": answer3
    }}
    logs.append(subtask_desc3)
```

Debate:
```python
    debate_instruction_5 = "Sub-task 5: Based on the output of Sub-task 4, convert [intermediate output] into [specific format] and calculate [the final answer]"
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_5 = self.max_round
    
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    
    subtask_desc5 = {{
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4",
        "agent_collaboration": "Debate"
    }}
    
    for r in range(N_max_5):
        # N_max_5 rounds of debating
        for i, agent in enumerate(debate_agents_5):
            # Each agent proposes its solution
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], 
                                           debate_instruction_5, r, is_sub_task=True)
            else:
                # Generate next solution based on comments and counter-arguments from other debaters
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            
            agents.append(f"Debate agent {{agent.id}}, round {{r}}, converting [intermediate output] and calculating [final output], thinking: {{thinking5.content}}; answer: {{answer_5.content}}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    
    # Final decision agent makes final decision
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], 
                                                 "Sub-task 5: Make final decision on [final output].", 
                                                 is_sub_task=True)
    agents.append(f"Final Decision agent, calculating [final output], thinking: {{thinking5.content}}; answer: {{answer5.content}}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {{thinking5.content}}; answer - {{answer5.content}}")
    subtask_desc5['response'] = {{
        "thinking": thinking5,
        "answer": answer5
    }}
    logs.append(subtask_desc5)
```
    """