async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Identify and define each listed issue in genomics data analysis—mutually incompatible data formats, the 'chr' / 'no chr' confusion, reference assembly mismatch, and incorrect ID conversion—to establish a clear, precise understanding of their nature and context."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identify and define genomics data analysis issues, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    cot_sc_instruction_2 = "Sub-task 2: Analyze the specific impact of each defined issue on generating difficult-to-spot erroneous results in genomics data analysis, explicitly linking how each issue can cause subtle or hidden errors, based on the definitions from Sub-task 1. Provide multiple perspectives to ensure thorough reasoning."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyze impact of genomics issues, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_reflect_instruction_3a = "Sub-task 3a: Collect and summarize quantitative evidence from peer-reviewed literature or reputable error-tracking databases regarding the frequency and prevalence of each issue as a source of difficult-to-spot errors in genomics data analysis. Provide citations or data references."
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_reflect_instruction_3a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking1, answer1, thinking2, answer2], cot_reflect_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, collect quantitative evidence on frequency of genomics issues, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    cot_reflect_instruction_3b = "Sub-task 3b: Tabulate and rank the issues based on the collected frequency data, identifying the top two or three most common sources of difficult-to-spot erroneous results, supported by citations or data references."
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3b = self.max_round
    cot_inputs_3b = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3a, answer3a]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_reflect_instruction_3b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, tabulate and rank genomics issues by frequency, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max_3b):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "Please review the tabulation and ranking of frequency data and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_reflect_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining tabulation and ranking, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    debate_instruction_4 = "Sub-task 4: Map the ranked issues to the provided multiple-choice options and select the option (A, B, C, or D) that best corresponds to the most common sources of difficult-to-spot erroneous results, ensuring the choice is evidence-based and justified. Debate the merits of each option based on the evidence."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3b", "answer of subtask 3b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3b, answer3b], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3b, answer3b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mapping ranked issues to options, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the most common sources of difficult-to-spot erroneous results in genomics data analysis, selecting the best multiple-choice option based on evidence.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on multiple-choice answer, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs