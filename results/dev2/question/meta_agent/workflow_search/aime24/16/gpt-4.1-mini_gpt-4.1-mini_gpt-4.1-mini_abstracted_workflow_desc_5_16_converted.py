async def forward_16(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Translate the given geometric conditions (IA perpendicular to OI, circumradius R=13, inradius r=6) "
        "into algebraic and geometric relations involving the triangle's sides, angles, and coordinates of points O and I. "
        "Express OI and IA in terms of R, r, and angle A using classical formulas."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, translating geometric conditions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Derive explicit algebraic relations involving sides AB, AC, BC and elements O, I from the perpendicularity condition IA ⟂ OI "
        "and the known radii (R=13, r=6), incorporating classical identities such as OI² = R(R - 2r) and IA² = R² - OI²."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, deriving algebraic relations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_debate_instruction_3a = (
        "Sub-task 3a: Using the relations from Stage 1, rigorously prove or disprove that the triangle is isosceles with AB = AC. "
        "This must be done via algebraic or geometric methods without assuming symmetry. Provide a formal proof or counterexample."
    )
    N_sc_3a = self.max_sc
    cot_agents_3a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.7) for _ in range(N_sc_3a)]
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.7) for role in self.debate_role]
    possible_answers_3a = []
    thinkingmapping_3a = {}
    answermapping_3a = {}
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_sc_debate_instruction_3a,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT | Debate"
    }
    for i in range(N_sc_3a):
        thinking3a, answer3a = await cot_agents_3a[i]([taskInfo, thinking2, answer2], cot_sc_debate_instruction_3a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3a[i].id}, proving/disproving isosceles, thinking: {thinking3a.content}; answer: {answer3a.content}")
        possible_answers_3a.append(answer3a.content)
        thinkingmapping_3a[answer3a.content] = thinking3a
        answermapping_3a[answer3a.content] = answer3a
    answer3a_content = Counter(possible_answers_3a).most_common(1)[0][0]
    thinking3a = thinkingmapping_3a[answer3a_content]
    answer3a = answermapping_3a[answer3a_content]

    debate_rounds_3a = self.max_round
    for r in range(debate_rounds_3a):
        new_thinking = []
        new_answer = []
        for agent in debate_agents_3a:
            if r == 0:
                t, a = await agent([taskInfo, thinking3a, answer3a], cot_sc_debate_instruction_3a, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3a, answer3a] + new_thinking + new_answer
                t, a = await agent(inputs, cot_sc_debate_instruction_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating isosceles proof, thinking: {t.content}; answer: {a.content}")
            new_thinking.append(t)
            new_answer.append(a)
        if any(ans.content.lower() in ["proven", "true", "yes"] for ans in new_answer):
            break
        thinking3a = new_thinking[-1]
        answer3a = new_answer[-1]

    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])

    is_isosceles = False
    if any(keyword in answer3a.content.lower() for keyword in ["proven", "true", "yes"]):
        is_isosceles = True

    if is_isosceles:
        cot_sc_instruction_3b = (
            "Sub-task 3b: Given the triangle is isosceles (AB = AC), solve for angle A and side lengths AB, AC, BC using the known formulas: "
            "IA = r / sin(A/2), OI² = R(R - 2r), and law of sines. Compute cos A and other necessary parameters step-by-step."
        )
        N_sc_3b = self.max_sc
        cot_agents_3b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.3) for _ in range(N_sc_3b)]
        possible_answers_3b = []
        thinkingmapping_3b = {}
        answermapping_3b = {}
        subtask_desc3b = {
            "subtask_id": "subtask_3b",
            "instruction": cot_sc_instruction_3b,
            "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(N_sc_3b):
            thinking3b, answer3b = await cot_agents_3b[i]([taskInfo, thinking3a, answer3a], cot_sc_instruction_3b, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_3b[i].id}, solving isosceles case, thinking: {thinking3b.content}; answer: {answer3b.content}")
            possible_answers_3b.append(answer3b.content)
            thinkingmapping_3b[answer3b.content] = thinking3b
            answermapping_3b[answer3b.content] = answer3b
        answer3b_content = Counter(possible_answers_3b).most_common(1)[0][0]
        thinking3b = thinkingmapping_3b[answer3b_content]
        answer3b = answermapping_3b[answer3b_content]
        sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
        subtask_desc3b['response'] = {
            "thinking": thinking3b,
            "answer": answer3b
        }
        logs.append(subtask_desc3b)
        print("Step 3b: ", sub_tasks[-1])

        cot_reflect_instruction_3d = (
            "Sub-task 3d: Perform a mandatory 'Check Assumptions' verification step immediately after computing cos A or side lengths. "
            "Explicitly confirm that no unproven symmetry or side-length equality assumptions are made. If assumptions exist, provide proofs or counterexamples; "
            "otherwise, flag the solution as assumption-free."
        )
        cot_agent_3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        critic_agent_3d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
        N_max_3d = self.max_round
        cot_inputs_3d = [taskInfo, thinking3b, answer3b]
        subtask_desc3d = {
            "subtask_id": "subtask_3d",
            "instruction": cot_reflect_instruction_3d,
            "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
            "agent_collaboration": "Reflexion | Debate"
        }
        thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, 0, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, checking assumptions in isosceles case, thinking: {thinking3d.content}; answer: {answer3d.content}")
        for i in range(N_max_3d):
            feedback, correct = await critic_agent_3d([taskInfo, thinking3d, answer3d], "please verify assumptions and provide proofs or counterexamples.", i, is_sub_task=True)
            agents.append(f"Critic agent {critic_agent_3d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
            if correct.content == "True":
                break
            cot_inputs_3d.extend([thinking3d, answer3d, feedback])
            thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, i + 1, is_sub_task=True)
            agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, refining assumption check, thinking: {thinking3d.content}; answer: {answer3d.content}")
        sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d.content}; answer - {answer3d.content}")
        subtask_desc3d['response'] = {
            "thinking": thinking3d,
            "answer": answer3d
        }
        logs.append(subtask_desc3d)
        print("Step 3d: ", sub_tasks[-1])

    else:
        cot_sc_debate_instruction_3c = (
            "Sub-task 3c: Since the triangle is not isosceles, analyze alternative triangle configurations that satisfy the given conditions. "
            "Explore possible values of angles and sides consistent with IA ⟂ OI, R=13, and r=6, using classical geometric relations and algebraic manipulation."
        )
        N_sc_3c = self.max_sc
        cot_agents_3c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.7) for _ in range(N_sc_3c)]
        debate_agents_3c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.7) for role in self.debate_role]
        possible_answers_3c = []
        thinkingmapping_3c = {}
        answermapping_3c = {}
        subtask_desc3c = {
            "subtask_id": "subtask_3c",
            "instruction": cot_sc_debate_instruction_3c,
            "context": ["user query", "thinking of subtask 3a", "answer of subtask 3a"],
            "agent_collaboration": "SC_CoT | Debate"
        }
        for i in range(N_sc_3c):
            thinking3c, answer3c = await cot_agents_3c[i]([taskInfo, thinking3a, answer3a], cot_sc_debate_instruction_3c, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_3c[i].id}, analyzing alternative configurations, thinking: {thinking3c.content}; answer: {answer3c.content}")
            possible_answers_3c.append(answer3c.content)
            thinkingmapping_3c[answer3c.content] = thinking3c
            answermapping_3c[answer3c.content] = answer3c
        answer3c_content = Counter(possible_answers_3c).most_common(1)[0][0]
        thinking3c = thinkingmapping_3c[answer3c_content]
        answer3c = answermapping_3c[answer3c_content]

        debate_rounds_3c = self.max_round
        for r in range(debate_rounds_3c):
            new_thinking = []
            new_answer = []
            for agent in debate_agents_3c:
                if r == 0:
                    t, a = await agent([taskInfo, thinking3c, answer3c], cot_sc_debate_instruction_3c, r, is_sub_task=True)
                else:
                    inputs = [taskInfo, thinking3c, answer3c] + new_thinking + new_answer
                    t, a = await agent(inputs, cot_sc_debate_instruction_3c, r, is_sub_task=True)
                agents.append(f"Debate agent {agent.id}, round {r}, debating alternative configurations, thinking: {t.content}; answer: {a.content}")
                new_thinking.append(t)
                new_answer.append(a)
            thinking3c = new_thinking[-1]
            answer3c = new_answer[-1]

        sub_tasks.append(f"Sub-task 3c output: thinking - {thinking3c.content}; answer - {answer3c.content}")
        subtask_desc3c['response'] = {
            "thinking": thinking3c,
            "answer": answer3c
        }
        logs.append(subtask_desc3c)
        print("Step 3c: ", sub_tasks[-1])

        cot_reflect_instruction_3d = (
            "Sub-task 3d: Perform a mandatory 'Check Assumptions' verification step immediately after computing cos A or side lengths. "
            "Explicitly confirm that no unproven symmetry or side-length equality assumptions are made. If assumptions exist, provide proofs or counterexamples; "
            "otherwise, flag the solution as assumption-free."
        )
        cot_agent_3d = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
        critic_agent_3d = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
        N_max_3d = self.max_round
        cot_inputs_3d = [taskInfo, thinking3c, answer3c]
        subtask_desc3d = {
            "subtask_id": "subtask_3d",
            "instruction": cot_reflect_instruction_3d,
            "context": ["user query", "thinking of subtask 3c", "answer of subtask 3c"],
            "agent_collaboration": "Reflexion | Debate"
        }
        thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, 0, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, checking assumptions in alternative case, thinking: {thinking3d.content}; answer: {answer3d.content}")
        for i in range(N_max_3d):
            feedback, correct = await critic_agent_3d([taskInfo, thinking3d, answer3d], "please verify assumptions and provide proofs or counterexamples.", i, is_sub_task=True)
            agents.append(f"Critic agent {critic_agent_3d.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
            if correct.content == "True":
                break
            cot_inputs_3d.extend([thinking3d, answer3d, feedback])
            thinking3d, answer3d = await cot_agent_3d(cot_inputs_3d, cot_reflect_instruction_3d, i + 1, is_sub_task=True)
            agents.append(f"Reflexion CoT agent {cot_agent_3d.id}, refining assumption check, thinking: {thinking3d.content}; answer: {answer3d.content}")
        sub_tasks.append(f"Sub-task 3d output: thinking - {thinking3d.content}; answer - {answer3d.content}")
        subtask_desc3d['response'] = {
            "thinking": thinking3d,
            "answer": answer3d
        }
        logs.append(subtask_desc3d)
        print("Step 3d: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Calculate the product AB * AC using the parameters and side lengths found in Stage 2. "
        "Combine all numeric values carefully, ensuring consistency with the verified assumptions and geometric constraints."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3d", "answer of subtask 3d"],
        "agent_collaboration": "SC_CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3d, answer3d], cot_sc_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating product AB*AC, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Critically verify the final answer for AB * AC against all given conditions (IA ⟂ OI, R=13, r=6) and the assumption checks. "
        "In the Debate phase, actively challenge any lingering assumptions, explore alternative configurations if needed, and confirm the correctness and uniqueness of the solution. "
        "Return the final verified numeric value alongside the verification report."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.7) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion | Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying final answer, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the verified value of AB * AC.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final verified value, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
