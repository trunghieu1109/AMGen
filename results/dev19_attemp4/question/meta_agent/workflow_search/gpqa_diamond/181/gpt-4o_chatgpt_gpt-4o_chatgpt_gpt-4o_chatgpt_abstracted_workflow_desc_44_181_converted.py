async def forward_181(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_0_1 = "Sub-task 1: Extract and summarize the given information about the Mott-Gurney equation, including its parameters and context."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing Mott-Gurney equation, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {
        "thinking": thinking_0_1,
        "answer": answer_0_1
    }
    logs.append(subtask_desc_0_1)
    
    cot_sc_instruction_0_2 = "Sub-task 2: Analyze the relationships between the components of the Mott-Gurney equation, focusing on how the parameters interact and the constraints of the equation."
    N = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", "thinking of subtask 0_1", "answer of subtask 0_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, analyzing relationships, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2)
        possible_thinkings_0_2.append(thinking_0_2)
    final_instr_0_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the relationships in the Mott-Gurney equation."
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_0_2 + possible_answers_0_2, 
                                                 "Sub-task 0_2: Synthesize and choose the most consistent answer for the relationships in the Mott-Gurney equation" + final_instr_0_2, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {
        "thinking": thinking_0_2,
        "answer": answer_0_2
    }
    logs.append(subtask_desc_0_2)
    
    debate_instruction_1_1 = "Sub-task 1: Evaluate the validity of each statement by comparing the conditions described in the choices with the constraints and assumptions of the Mott-Gurney equation."
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                   model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", "thinking of subtask 0_2", "answer of subtask 0_2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking_1_1, answer_1_1 = await agent([taskInfo, thinking_0_2, answer_0_2], 
                                           debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking_0_2, answer_0_2] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking_1_1, answer_1_1 = await agent(input_infos_1_1, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating validity, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
            all_thinking_1_1[r].append(thinking_1_1)
            all_answer_1_1[r].append(answer_1_1)
    final_instr_1_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_2, answer_0_2] + all_thinking_1_1[-1] + all_answer_1_1[-1], 
                                                 "Sub-task 1_1: Evaluate and finalize the validity of statements" + final_instr_1_1, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {
        "thinking": thinking_1_1,
        "answer": answer_1_1
    }
    logs.append(subtask_desc_1_1)
    
    cot_instruction_2_1 = "Sub-task 1: Identify and classify the correct statement about the validity of the Mott-Gurney equation based on the analysis and evaluation from previous stages."
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_1, answer_1_1], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, classifying correct statement, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {
        "thinking": thinking_2_1,
        "answer": answer_2_1
    }
    logs.append(subtask_desc_2_1)
    
    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs