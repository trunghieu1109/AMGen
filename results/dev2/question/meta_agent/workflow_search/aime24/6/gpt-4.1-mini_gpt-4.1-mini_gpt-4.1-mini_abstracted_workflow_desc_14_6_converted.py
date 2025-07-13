async def forward_6(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify the set B of rectangular boxes with dimensions (x,y,z) such that the surface area is 54 "
        "and the volume is 23. Explicitly state the constraints: 2(xy + yz + zx) = 54 and xyz = 23. "
        "Provide reasoning about the feasibility and nature of these boxes."
    )
    subtask_id_1 = "subtask_1"
    print(f"Logging before agent call: subtask_id={subtask_id_1}, instruction={cot_instruction_1}, context=['user query'], agent_collaboration=CoT")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying and verifying boxes, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1 = {
        "subtask_id": subtask_id_1,
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT",
        "response": {"thinking": thinking_1, "answer": answer_1}
    }
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Derive the formula for the squared radius r^2 of the smallest sphere containing a box with dimensions (x,y,z), "
        "showing that r^2 = (x^2 + y^2 + z^2)/4. Clarify that the radius depends on the maximum diagonal length among all boxes in B. "
        "Provide detailed derivation and validation."
    )
    subtask_id_2 = "subtask_2"
    print(f"Logging before agent call: subtask_id={subtask_id_2}, instruction={cot_instruction_2}, context=['user query', thinking_1.content, answer_1.content], agent_collaboration=CoT")
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await cot_agent_2([taskInfo, thinking_1, answer_1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, deriving algebraic relations and radius formula, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2 = {
        "subtask_id": subtask_id_2,
        "instruction": cot_instruction_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT",
        "response": {"thinking": thinking_2, "answer": answer_2}
    }
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3a = (
        "Sub-task 3a: Formulate the optimization problem to find the maximum possible value of x^2 + y^2 + z^2 "
        "subject to the constraints 2(xy + yz + zx) = 54 and xyz = 23. Explicitly state that maximizing x^2 + y^2 + z^2 "
        "corresponds to finding the smallest sphere radius containing all boxes."
    )
    cot_sc_instruction_3b = (
        "Sub-task 3b: Reduce the optimization problem to the symmetric case x = y = a, derive the cubic equation a^3 - 27a + 46 = 0, "
        "and solve for all real roots a."
    )
    cot_sc_instruction_3c = (
        "Sub-task 3c: For each real root a found, compute the corresponding z = 23 / a^2, then calculate r^2 = (2a^2 + z^2)/4. "
        "Identify the maximum r^2 among these candidates."
    )

    N = self.max_sc
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    cot_agents_3c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]

    possible_answers_3c = []
    thinkingmapping_3c = {}
    answermapping_3c = {}

    subtask_desc_3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_instruction_3a,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content],
        "agent_collaboration": "SC_CoT"
    }
    subtask_desc_3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_sc_instruction_3b,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content],
        "agent_collaboration": "SC_CoT"
    }
    subtask_desc_3c = {
        "subtask_id": "subtask_3c",
        "instruction": cot_sc_instruction_3c,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content],
        "agent_collaboration": "SC_CoT"
    }

    # Sub-task 3a
    print(f"Logging before agent calls: subtask_id=subtask_3a, instruction={cot_sc_instruction_3a}, context=['user query', thinking_1.content, answer_1.content, thinking_2.content, answer_2.content], agent_collaboration=SC_CoT")
    thinking_3a_list = []
    answer_3a_list = []
    for i in range(N):
        thinking_3a, answer_3a = await cot_agents_3a[i]([taskInfo, thinking_1, answer_1, thinking_2, answer_2], cot_sc_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, formulating optimization problem, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
        thinking_3a_list.append(thinking_3a)
        answer_3a_list.append(answer_3a)
    # Choose most common answer (if applicable) or first
    answer_3a_content = Counter([a.content for a in answer_3a_list]).most_common(1)[0][0]
    thinking_3a = next(t for t,a in zip(thinking_3a_list, answer_3a_list) if a.content == answer_3a_content)
    answer_3a = next(a for a in answer_3a_list if a.content == answer_3a_content)
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 3a: ", sub_tasks[-1])

    # Sub-task 3b
    print(f"Logging before agent calls: subtask_id=subtask_3b, instruction={cot_sc_instruction_3b}, context=['user query', thinking_1.content, answer_1.content, thinking_2.content, answer_2.content, thinking_3a.content, answer_3a.content], agent_collaboration=SC_CoT")
    thinking_3b_list = []
    answer_3b_list = []
    for i in range(N):
        thinking_3b, answer_3b = await cot_agents_3b[i]([taskInfo, thinking_1, answer_1, thinking_2, answer_2, thinking_3a, answer_3a], cot_sc_instruction_3b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, reducing to symmetric case and solving cubic, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
        thinking_3b_list.append(thinking_3b)
        answer_3b_list.append(answer_3b)
    answer_3b_content = Counter([a.content for a in answer_3b_list]).most_common(1)[0][0]
    thinking_3b = next(t for t,a in zip(thinking_3b_list, answer_3b_list) if a.content == answer_3b_content)
    answer_3b = next(a for a in answer_3b_list if a.content == answer_3b_content)
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 3b: ", sub_tasks[-1])

    # Sub-task 3c
    print(f"Logging before agent calls: subtask_id=subtask_3c, instruction={cot_sc_instruction_3c}, context=['user query', thinking_1.content, answer_1.content, thinking_2.content, answer_2.content, thinking_3b.content, answer_3b.content], agent_collaboration=SC_CoT")
    for i in range(N):
        thinking_3c, answer_3c = await cot_agents_3c[i]([taskInfo, thinking_1, answer_1, thinking_2, answer_2, thinking_3b, answer_3b], cot_sc_instruction_3c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3c[i].id}, evaluating candidates and computing max r^2, thinking: {thinking_3c.content}; answer: {answer_3c.content}")
        possible_answers_3c.append(answer_3c.content)
        thinkingmapping_3c[answer_3c.content] = thinking_3c
        answermapping_3c[answer_3c.content] = answer_3c
    answer_3c_content = Counter(possible_answers_3c).most_common(1)[0][0]
    thinking_3c = thinkingmapping_3c[answer_3c_content]
    answer_3c = answermapping_3c[answer_3c_content]
    sub_tasks.append(f"Sub-task 3c output: thinking - {thinking_3c.content}; answer - {answer_3c.content}")
    subtask_desc_3c['response'] = {"thinking": thinking_3c, "answer": answer_3c}
    logs.append(subtask_desc_3c)
    print("Step 3c: ", sub_tasks[-1])

    cot_reflect_instruction_3d = (
        "Sub-task 3d: Verify that the computed maximum radius r indeed contains all boxes in B by analyzing boundary cases or using geometric reasoning. "
        "Confirm the correctness of the optimization direction and solution feasibility."
    )
    subtask_id_3d = "subtask_3d"
    print(f"Logging before agent call: subtask_id={subtask_id_3d}, instruction={cot_reflect_instruction_3d}, context=['user query', thinking_1.content, answer_1.content, thinking_2.content, answer_2.content, thinking_3c.content, answer_3c.content], agent_collaboration=Reflexion")
    cot_agent_3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3d = self.max_round
    cot_inputs_3d = [taskInfo, thinking_1, answer_1, thinking_2, answer_2, thinking_3c, answer_3c]
    thinking_3d, answer_3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, verifying max radius and solution feasibility, thinking: {thinking_3d.content}; answer: {answer_3d.content}")
    for i in range(N_max_3d):
        feedback, correct = await critic_agent_3d([taskInfo, thinking_3d, answer_3d],
                                                "Please review the verification of the maximum radius and provide limitations or correctness.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_3d.extend([thinking_3d, answer_3d, feedback])
        thinking_3d, answer_3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, refining verification, thinking: {thinking_3d.content}; answer: {answer_3d.content}")
    sub_tasks.append(f"Sub-task 3d output: thinking - {thinking_3d.content}; answer - {answer_3d.content}")
    subtask_desc_3d = {
        "subtask_id": subtask_id_3d,
        "instruction": cot_reflect_instruction_3d,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content, thinking_3c.content, answer_3c.content],
        "agent_collaboration": "Reflexion",
        "response": {"thinking": thinking_3d, "answer": answer_3d}
    }
    logs.append(subtask_desc_3d)
    print("Step 3d: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Simplify the fraction r^2 = p/q obtained from the maximum diagonal squared to lowest terms where p and q are relatively prime positive integers. "
        "Compute and return the final answer p + q along with the verification result from Sub-task 3d."
    )
    subtask_id_4 = "subtask_4"
    print(f"Logging before agent call: subtask_id={subtask_id_4}, instruction={cot_instruction_4}, context=['user query', thinking_1.content, answer_1.content, thinking_2.content, answer_2.content, thinking_3d.content, answer_3d.content], agent_collaboration=CoT")
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_4, answer_4 = await cot_agent_4([taskInfo, thinking_1, answer_1, thinking_2, answer_2, thinking_3d, answer_3d], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, simplifying fraction and computing p+q, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4 = {
        "subtask_id": subtask_id_4,
        "instruction": cot_instruction_4,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content, thinking_3d.content, answer_3d.content],
        "agent_collaboration": "CoT",
        "response": {"thinking": thinking_4, "answer": answer_4}
    }
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Conduct a debate and reflexion stage focusing on validating the entire solution pipeline: confirm the geometric interpretation, "
        "verify the optimization direction (maximization), check algebraic manipulations, and ensure the final fraction simplification and arithmetic are correct. "
        "Assign roles for mathematical validation and geometric reasoning to catch any conceptual or computational errors."
    )
    subtask_id_5 = "subtask_5"
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": subtask_id_5,
        "instruction": debate_instruction_5,
        "context": ["user query", thinking_1.content, answer_1.content, thinking_2.content, answer_2.content, thinking_3d.content, answer_3d.content, thinking_4.content, answer_4.content],
        "agent_collaboration": "Debate"
    }
    print(f"Logging before debate agents calls: subtask_id={subtask_id_5}, instruction={debate_instruction_5}, context=[user query and all previous thinking and answers], agent_collaboration=Debate")
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating entire solution, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1],
                                                      "Sub-task 5: Make final decision on the validated solution and final answer.",
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on solution correctness and final answer, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_5, answer_5, sub_tasks, agents)
    return final_answer, logs
