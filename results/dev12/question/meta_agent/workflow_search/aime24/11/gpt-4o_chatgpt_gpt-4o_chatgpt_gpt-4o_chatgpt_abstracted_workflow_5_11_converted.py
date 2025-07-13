async def forward_11(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs =  []
    
    cot_instruction_0_1 = "Sub-task 1: Identify the total number of moves required to traverse from the lower left corner to the upper right corner on an 8x8 grid, considering the grid's dimensions and the path length."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing total moves, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {
        "thinking": thinking_0_1,
        "answer": answer_0_1
    }
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_0_2 = "Sub-task 2: Determine the number of horizontal and vertical moves needed to reach the opposite corner, given the grid's dimensions."
    N_0_2 = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_0_2)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", "thinking of subtask 0_1", "answer of subtask 0_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_0_2):
        thinking_0_2, answer_0_2 = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, determining moves, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
        possible_answers_0_2.append(answer_0_2)
        possible_thinkings_0_2.append(thinking_0_2)
    final_instr_0_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the number of moves."
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_answers_0_2 + possible_thinkings_0_2, 
                                                 "Sub-task 0_2: Synthesize and choose the most consistent answer for moves" + final_instr_0_2, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {
        "thinking": thinking_0_2,
        "answer": answer_0_2
    }
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_1_1 = "Sub-task 1: Define the concept of a direction change in the context of grid paths and establish the conditions for a path to change direction exactly four times."
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", "thinking of subtask 0_2", "answer of subtask 0_2"],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, defining direction change, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {
        "thinking": thinking_1_1,
        "answer": answer_1_1
    }
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_1_2 = "Sub-task 2: Enumerate the possible sequences of moves that result in exactly four direction changes, considering the constraints identified in the previous subtasks."
    N_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                              model=self.node_model, temperature=0.5) for _ in range(N_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", "thinking of subtask 1_1", "answer of subtask 1_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1_2):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, enumerating sequences, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_instr_1_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the sequences."
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                         model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, 
                                                 "Sub-task 1_2: Synthesize and choose the most consistent answer for sequences" + final_instr_1_2, 
                                                 is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {
        "thinking": thinking_1_2,
        "answer": answer_1_2
    }
    logs.append(subtask_desc_1_2)
    print("Step 4: ", sub_tasks[-1])
    
    reflect_inst_2_1 =  "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = "Sub-task 3: Calculate the number of valid paths that meet the criteria of having exactly four direction changes, using combinatorial methods." + reflect_inst_2_1
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                            model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", "thinking of subtask 1_2", "answer of subtask 1_2"],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, calculating valid paths, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    critic_inst_2_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], 
                                       "Please review and provide the limitations of provided solutions" + critic_inst_2_1, 
                                       i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining valid paths, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {
        "thinking": thinking_2_1,
        "answer": answer_2_1
    }
    logs.append(subtask_desc_2_1)
    print("Step 5: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs