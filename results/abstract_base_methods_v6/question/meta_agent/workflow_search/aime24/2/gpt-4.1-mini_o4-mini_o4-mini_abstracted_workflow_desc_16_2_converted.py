async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Understand the problem setup: a regular octagon with 8 vertices, each independently colored red or blue with equal probability (1/2 each), and the event that there exists a rotation of the octagon such that all blue vertices map to positions originally colored red."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understand problem setup, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Identify and describe the group of rotations of the regular octagon (the cyclic group of order 8), and characterize how each rotation permutes the vertices, including the cycle structure of each rotation."
    N2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identify rotation group and vertex permutation, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Sub-task 3: Formulate the event A_k for each rotation k (k=0 to 7) that the coloring satisfies the condition that all blue vertices map under rotation k to vertices originally red, expressing this event in terms of the coloring and the rotation action on vertices."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, formulate event A_k for rotations, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the event formulation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining event formulation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_reflect_instruction_4 = "Sub-task 4: For each rotation k (k=0 to 7), determine the combinatorial condition on the coloring for event A_k to hold, using the cycle decomposition of the rotation to identify constraints on vertex colors (e.g., fixed points, orbits), and compute the number of colorings satisfying A_k."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, determine coloring condition per rotation, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "please review the coloring condition per rotation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining coloring condition, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5a = "Sub-task 5a: Compute the size |A_k| of each event A_k (the number of colorings fixed by rotation k under the blue-to-red mapping condition), with detailed combinatorial justification based on the cycle structure and coloring constraints."
    cot_agents_5a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_5a = []
    thinkingmapping_5a = {}
    answermapping_5a = {}
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking5a, answer5a = await cot_agents_5a[i]([taskInfo, thinking4, answer4], cot_instruction_5a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5a[i].id}, compute size |A_k| for each rotation, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a.content)
        thinkingmapping_5a[answer5a.content] = thinking5a
        answermapping_5a[answer5a.content] = answer5a
    answer5a_content = Counter(possible_answers_5a).most_common(1)[0][0]
    thinking5a = thinkingmapping_5a[answer5a_content]
    answer5a = answermapping_5a[answer5a_content]
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_instruction_5b = "Sub-task 5b: Compute the sizes of pairwise intersections |A_i ∩ A_j| for all distinct pairs of rotations i, j, by analyzing combined constraints from both rotations and counting colorings satisfying both events."
    cot_agents_5b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_5b = []
    thinkingmapping_5b = {}
    answermapping_5b = {}
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking5b, answer5b = await cot_agents_5b[i]([taskInfo, thinking5a, answer5a], cot_instruction_5b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5b[i].id}, compute pairwise intersections |A_i ∩ A_j|, thinking: {thinking5b.content}; answer: {answer5b.content}")
        possible_answers_5b.append(answer5b.content)
        thinkingmapping_5b[answer5b.content] = thinking5b
        answermapping_5b[answer5b.content] = answer5b
    answer5b_content = Counter(possible_answers_5b).most_common(1)[0][0]
    thinking5b = thinkingmapping_5b[answer5b_content]
    answer5b = answermapping_5b[answer5b_content]
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_instruction_5c = "Sub-task 5c: Compute sizes of higher-order intersections (triplets, quadruplets, etc.) |A_i1 ∩ A_i2 ∩ ...| as needed for the inclusion-exclusion principle, ensuring all non-empty intersections are accounted for with combinatorial justification."
    cot_agents_5c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_5c = []
    thinkingmapping_5c = {}
    answermapping_5c = {}
    subtask_desc5c = {
        "subtask_id": "subtask_5c",
        "instruction": cot_instruction_5c,
        "context": ["user query", "thinking of subtask 5b", "answer of subtask 5b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking5c, answer5c = await cot_agents_5c[i]([taskInfo, thinking5b, answer5b], cot_instruction_5c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5c[i].id}, compute higher-order intersections for inclusion-exclusion, thinking: {thinking5c.content}; answer: {answer5c.content}")
        possible_answers_5c.append(answer5c.content)
        thinkingmapping_5c[answer5c.content] = thinking5c
        answermapping_5c[answer5c.content] = answer5c
    answer5c_content = Counter(possible_answers_5c).most_common(1)[0][0]
    thinking5c = thinkingmapping_5c[answer5c_content]
    answer5c = answermapping_5c[answer5c_content]
    sub_tasks.append(f"Sub-task 5c output: thinking - {thinking5c.content}; answer - {answer5c.content}")
    subtask_desc5c['response'] = {"thinking": thinking5c, "answer": answer5c}
    logs.append(subtask_desc5c)
    print("Step 5c: ", sub_tasks[-1])
    
    cot_instruction_5d = "Sub-task 5d: Apply the inclusion-exclusion principle step-by-step using the computed sizes of all intersections to find the total number of colorings in the union |∪_k=0^7 A_k|, i.e., colorings for which there exists at least one rotation satisfying the blue-to-red mapping condition."
    cot_agent_5d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5d = self.max_round
    cot_inputs_5d = [taskInfo, thinking5a, answer5a, thinking5b, answer5b, thinking5c, answer5c]
    subtask_desc5d = {
        "subtask_id": "subtask_5d",
        "instruction": cot_instruction_5d,
        "context": ["user query", "thinking of subtask 5a", "answer of subtask 5a", "thinking of subtask 5b", "answer of subtask 5b", "thinking of subtask 5c", "answer of subtask 5c"],
        "agent_collaboration": "Reflexion"
    }
    thinking5d, answer5d = await cot_agent_5d(cot_inputs_5d, cot_instruction_5d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5d.id}, apply inclusion-exclusion principle step-by-step, thinking: {thinking5d.content}; answer: {answer5d.content}")
    for i in range(N_max_5d):
        feedback, correct = await critic_agent_5d([taskInfo, thinking5d, answer5d], "please review the inclusion-exclusion calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5d.extend([thinking5d, answer5d, feedback])
        thinking5d, answer5d = await cot_agent_5d(cot_inputs_5d, cot_instruction_5d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5d.id}, refining inclusion-exclusion calculation, thinking: {thinking5d.content}; answer: {answer5d.content}")
    sub_tasks.append(f"Sub-task 5d output: thinking - {thinking5d.content}; answer - {answer5d.content}")
    subtask_desc5d['response'] = {"thinking": thinking5d, "answer": answer5d}
    logs.append(subtask_desc5d)
    print("Step 5d: ", sub_tasks[-1])
    
    cot_instruction_5e = "Sub-task 5e: Alternatively, verify the union size by complement counting: prove that any blue set of size ≥5 intersects all nonzero rotations, so only blue sets of size ≤4 can satisfy the condition, and sum the corresponding binomial coefficients to cross-check the inclusion-exclusion result."
    cot_agent_5e = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5e = [taskInfo, thinking5d, answer5d]
    subtask_desc5e = {
        "subtask_id": "subtask_5e",
        "instruction": cot_instruction_5e,
        "context": ["user query", "thinking of subtask 5d", "answer of subtask 5d"],
        "agent_collaboration": "CoT"
    }
    thinking5e, answer5e = await cot_agent_5e(cot_inputs_5e, cot_instruction_5e, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5e.id}, verify union size by complement counting, thinking: {thinking5e.content}; answer: {answer5e.content}")
    sub_tasks.append(f"Sub-task 5e output: thinking - {thinking5e.content}; answer - {answer5e.content}")
    subtask_desc5e['response'] = {"thinking": thinking5e, "answer": answer5e}
    logs.append(subtask_desc5e)
    print("Step 5e: ", sub_tasks[-1])
    
    cot_reflect_instruction_5f = "Sub-task 5f: Cross-validate the results from inclusion-exclusion and complement counting methods to confirm the correctness and consistency of the union size |∪ A_k|, resolving any discrepancies before proceeding."
    cot_agent_5f = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5f = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5f = self.max_round
    cot_inputs_5f = [taskInfo, thinking5d, answer5d, thinking5e, answer5e]
    subtask_desc5f = {
        "subtask_id": "subtask_5f",
        "instruction": cot_reflect_instruction_5f,
        "context": ["user query", "thinking of subtask 5d", "answer of subtask 5d", "thinking of subtask 5e", "answer of subtask 5e"],
        "agent_collaboration": "Reflexion"
    }
    thinking5f, answer5f = await cot_agent_5f(cot_inputs_5f, cot_reflect_instruction_5f, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5f.id}, cross-validate inclusion-exclusion and complement counting results, thinking: {thinking5f.content}; answer: {answer5f.content}")
    for i in range(N_max_5f):
        feedback, correct = await critic_agent_5f([taskInfo, thinking5f, answer5f], "please review the cross-validation and confirm consistency.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5f.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5f.extend([thinking5f, answer5f, feedback])
        thinking5f, answer5f = await cot_agent_5f(cot_inputs_5f, cot_reflect_instruction_5f, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5f.id}, refining cross-validation, thinking: {thinking5f.content}; answer: {answer5f.content}")
    sub_tasks.append(f"Sub-task 5f output: thinking - {thinking5f.content}; answer - {answer5f.content}")
    subtask_desc5f['response'] = {"thinking": thinking5f, "answer": answer5f}
    logs.append(subtask_desc5f)
    print("Step 5f: ", sub_tasks[-1])
    
    cot_reflect_instruction_6 = "Sub-task 6: Calculate the probability that the octagon coloring satisfies the event (exists a rotation mapping blue vertices to originally red vertices) by dividing the union size |∪ A_k| by the total number of colorings (2^8), and simplify the resulting fraction to lowest terms (m/n) with m and n relatively prime."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5f, answer5f]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_reflect_instruction_6,
        "context": ["user query", "thinking of subtask 5f", "answer of subtask 5f"],
        "agent_collaboration": "Reflexion"
    }
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, calculate and simplify probability fraction, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "please review the simplification and final fraction correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining simplification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_reflect_instruction_7 = "Sub-task 7: Compute and return the sum m + n as the final answer, where m/n is the simplified probability fraction obtained in Sub-task 6."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, compute sum m+n from simplified fraction, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback, correct = await critic_agent_7([taskInfo, thinking7, answer7], "please review the final sum computation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining final sum, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
