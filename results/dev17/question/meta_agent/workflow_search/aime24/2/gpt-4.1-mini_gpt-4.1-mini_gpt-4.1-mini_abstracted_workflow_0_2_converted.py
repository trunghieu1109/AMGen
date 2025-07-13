async def forward_2(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally define the problem setting by representing the octagon vertices as the set {0,1,...,7}, "
        "define a coloring as a function from vertices to {red, blue}, and represent rotations as elements of the cyclic group of order 8 acting on vertex indices. "
        "Emphasize the independence and equal probability (1/2) of coloring each vertex red or blue. Avoid attempting any enumeration or probability calculation at this stage."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formal problem definition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Express the condition that there exists a rotation r such that the set of blue vertices after rotation is a subset of the original red vertices. "
        "Translate this condition into a formal combinatorial constraint involving the coloring function and the rotation action. Avoid conflating this with counting or probability computations."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formalize rotation-subset condition, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Clarify the probabilistic model: state that each vertex is independently colored red or blue with probability 1/2, "
        "the total sample space size is 2^8=256, and the probability sought is the fraction of colorings satisfying the rotation-subset condition. "
        "Avoid any enumeration or counting here; focus solely on the probabilistic framework."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, clarify probabilistic model, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Analyze the structure of the rotation group acting on the octagon vertices. For each rotation r_k by k steps (k=1 to 7), "
        "explicitly compute and document the cycle decomposition of the vertex set under r_k. Emphasize correct gcd computations and cycle lengths, "
        "especially for rotations by 3 and 5 steps. Output a detailed table mapping each rotation to its cycle decomposition. Avoid skipping or approximating cycle structures."
    )
    N_sc = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, compute cycle decompositions, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent cycle decompositions for all rotations", is_sub_task=True)
    agents.append(f"Final Decision agent, cycle decomposition synthesis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5a = (
        "Sub-task 5a: For each rotation r_k (k=1 to 7), use the cycle decomposition from subtask_4 to enumerate and count the number of colorings fixed under the condition that the blue vertices after rotation are disjoint from the original blue vertices. "
        "Compute the exact size |A_k| of the set of colorings satisfying the condition for rotation r_k. Provide explicit numeric counts and document the counting process carefully, using cycle structures to simplify enumeration. Avoid vague or symbolic descriptions without numeric results."
    )
    cot_agents_5a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_5a = []
    possible_thinkings_5a = []
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_sc_instruction_5a,
        "context": ["user query", thinking3.content, thinking4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5a, answer5a = await cot_agents_5a[i]([taskInfo, thinking3, thinking4], cot_sc_instruction_5a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5a[i].id}, enumerate and count |A_k|, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a)
        possible_thinkings_5a.append(thinking5a)
    final_decision_agent_5a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5a, answer5a = await final_decision_agent_5a([taskInfo] + possible_thinkings_5a, "Sub-task 5a: Synthesize and choose the most consistent numeric counts |A_k| for all rotations", is_sub_task=True)
    agents.append(f"Final Decision agent, numeric counts |A_k| synthesis, thinking: {thinking5a.content}; answer: {answer5a.content}")
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])

    reflect_inst_5b = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_5b = (
        "Sub-task 5b: Verify the counts |A_k| obtained in subtask_5a by cross-validation methods such as partial brute-force enumeration or alternative combinatorial arguments. "
        "Document any discrepancies and resolve them. Avoid accepting counts without verification. " + reflect_inst_5b
    )
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5b = [taskInfo, thinking5a, answer5a]
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_reflect_instruction_5b,
        "context": ["user query", thinking5a.content, answer5a.content],
        "agent_collaboration": "Reflexion"
    }
    thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_reflect_instruction_5b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, verify |A_k| counts, thinking: {thinking5b.content}; answer: {answer5b.content}")
    critic_inst_5b = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback, correct = await critic_agent_5b([taskInfo, thinking5b], critic_inst_5b, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5b.id}, feedback on |A_k| verification, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5b.extend([thinking5b, feedback])
        thinking5b, answer5b = await cot_agent_5b(cot_inputs_5b, cot_reflect_instruction_5b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5b.id}, refine |A_k| verification, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])

    cot_sc_instruction_6a = (
        "Sub-task 6a: Compute all pairwise intersections |A_i ∩ A_j| for 1 ≤ i < j ≤ 7 by analyzing combined cycle structures or using enumeration techniques. "
        "Provide explicit numeric values for these intersections. Avoid skipping intersection computations or approximating them."
    )
    cot_agents_6a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_6a = []
    possible_thinkings_6a = []
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_sc_instruction_6a,
        "context": ["user query", thinking5b.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6a, answer6a = await cot_agents_6a[i]([taskInfo, thinking5b], cot_sc_instruction_6a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6a[i].id}, compute pairwise intersections, thinking: {thinking6a.content}; answer: {answer6a.content}")
        possible_answers_6a.append(answer6a)
        possible_thinkings_6a.append(thinking6a)
    final_decision_agent_6a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6a, answer6a = await final_decision_agent_6a([taskInfo] + possible_thinkings_6a, "Sub-task 6a: Synthesize and choose the most consistent numeric values for pairwise intersections", is_sub_task=True)
    agents.append(f"Final Decision agent, pairwise intersections synthesis, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])

    cot_sc_instruction_6b = (
        "Sub-task 6b: Compute higher-order intersections (triplets, quadruplets, etc.) |A_i ∩ A_j ∩ ...| as needed for the inclusion-exclusion principle. "
        "Provide explicit numeric values and document the methodology used to obtain these counts. Avoid incomplete or partial intersection computations."
    )
    cot_agents_6b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_6b = []
    possible_thinkings_6b = []
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_sc_instruction_6b,
        "context": ["user query", thinking6a.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6b, answer6b = await cot_agents_6b[i]([taskInfo, thinking6a], cot_sc_instruction_6b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6b[i].id}, compute higher-order intersections, thinking: {thinking6b.content}; answer: {answer6b.content}")
        possible_answers_6b.append(answer6b)
        possible_thinkings_6b.append(thinking6b)
    final_decision_agent_6b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6b, answer6b = await final_decision_agent_6b([taskInfo] + possible_thinkings_6b, "Sub-task 6b: Synthesize and choose the most consistent numeric values for higher-order intersections", is_sub_task=True)
    agents.append(f"Final Decision agent, higher-order intersections synthesis, thinking: {thinking6b.content}; answer: {answer6b.content}")
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])

    reflect_inst_6c = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_6c = (
        "Sub-task 6c: Apply the inclusion-exclusion principle step-by-step using the explicit numeric values of |A_k| and their intersections from subtasks 5 and 6a,b. "
        "Calculate the total number of favorable colorings that satisfy the rotation-subset condition. Document each term and intermediate sum clearly. Avoid jumping to conclusions or skipping steps in the aggregation. " + reflect_inst_6c
    )
    cot_agent_6c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6c = [taskInfo, thinking5b, thinking6a, thinking6b]
    subtask_desc6c = {
        "subtask_id": "subtask_6c",
        "instruction": cot_reflect_instruction_6c,
        "context": ["user query", thinking5b.content, thinking6a.content, thinking6b.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6c, answer6c = await cot_agent_6c(cot_inputs_6c, cot_reflect_instruction_6c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6c.id}, inclusion-exclusion aggregation, thinking: {thinking6c.content}; answer: {answer6c.content}")
    critic_inst_6c = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback, correct = await critic_agent_6c([taskInfo, thinking6c], critic_inst_6c, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6c.id}, feedback on inclusion-exclusion, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6c.extend([thinking6c, feedback])
        thinking6c, answer6c = await cot_agent_6c(cot_inputs_6c, cot_reflect_instruction_6c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6c.id}, refine inclusion-exclusion, thinking: {thinking6c.content}; answer: {answer6c.content}")
    sub_tasks.append(f"Sub-task 6c output: thinking - {thinking6c.content}; answer - {answer6c.content}")
    subtask_desc6c['response'] = {"thinking": thinking6c, "answer": answer6c}
    logs.append(subtask_desc6c)
    print("Step 6c: ", sub_tasks[-1])

    reflect_inst_6d = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_6d = (
        "Sub-task 6d: Cross-validate the total count of favorable colorings obtained from inclusion-exclusion with a brute-force enumeration over all 256 colorings, checking the rotation-subset condition directly. "
        "Report any discrepancies and reconcile them. Avoid accepting results without this verification step. " + reflect_inst_6d
    )
    cot_agent_6d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_6d = [taskInfo, thinking6c, answer6c]
    subtask_desc6d = {
        "subtask_id": "subtask_6d",
        "instruction": cot_reflect_instruction_6d,
        "context": ["user query", thinking6c.content, answer6c.content],
        "agent_collaboration": "Reflexion"
    }
    thinking6d, answer6d = await cot_agent_6d(cot_inputs_6d, cot_reflect_instruction_6d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6d.id}, brute-force verification, thinking: {thinking6d.content}; answer: {answer6d.content}")
    critic_inst_6d = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback, correct = await critic_agent_6d([taskInfo, thinking6d], critic_inst_6d, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6d.id}, feedback on brute-force verification, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6d.extend([thinking6d, feedback])
        thinking6d, answer6d = await cot_agent_6d(cot_inputs_6d, cot_reflect_instruction_6d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6d.id}, refine brute-force verification, thinking: {thinking6d.content}; answer: {answer6d.content}")
    sub_tasks.append(f"Sub-task 6d output: thinking - {thinking6d.content}; answer - {answer6d.content}")
    subtask_desc6d['response'] = {"thinking": thinking6d, "answer": answer6d}
    logs.append(subtask_desc6d)
    print("Step 6d: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Compute the probability as the ratio of favorable colorings (from subtask_6c or 6d) to total colorings (256). "
        "Simplify the fraction to lowest terms by computing gcd of numerator and denominator. Provide explicit values for m and n. Avoid skipping simplification or assuming coprimality without verification."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6c.content, answer6c.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6c, answer6c], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, compute and simplify probability fraction, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Calculate and output the final answer m + n, where m/n is the simplified probability fraction obtained in subtask_7. "
        "Ensure clarity and correctness in the final output. Avoid ambiguity or incomplete final statements."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking7.content, answer7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, compute final m+n, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
