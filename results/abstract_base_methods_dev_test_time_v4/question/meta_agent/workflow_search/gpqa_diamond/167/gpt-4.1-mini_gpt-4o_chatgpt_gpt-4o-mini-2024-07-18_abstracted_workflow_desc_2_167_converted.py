async def forward_167(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Define and clarify each listed issue in genomics data analysis (mutually incompatible data formats, the 'chr' / 'no chr' confusion, reference assembly mismatch, incorrect ID conversion) to establish a precise understanding of their nature, typical manifestations, and potential to cause errors." 
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, define and clarify genomics data analysis issues, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction_2 = "Sub-task 2: Identify common workflows and scenarios in genomics data analysis where each issue arises, emphasizing how these issues lead to erroneous or difficult-to-spot results, supported by concrete examples or case studies, based on the clarifications from Sub-task 1." 
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
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identify scenarios causing difficult-to-spot errors, thinking: {thinking2.content}; answer: {answer2.content}")
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
    cot_instruction_3a = "Sub-task 3a: Gather and summarize domain-specific evidence from literature, surveys, and bug reports quantifying the frequency and subtlety of each issue as a source of difficult-to-spot erroneous results in genomics data analysis, based on outputs from Sub-task 2." 
    cot_agent_3a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3a = {
        "subtask_id": "subtask_3a",
        "instruction": cot_instruction_3a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3a, answer3a = await cot_agent_3a([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3a.id}, gather domain evidence on frequency and subtlety, thinking: {thinking3a.content}; answer: {answer3a.content}")
    sub_tasks.append(f"Sub-task 3a output: thinking - {thinking3a.content}; answer - {answer3a.content}")
    subtask_desc3a['response'] = {
        "thinking": thinking3a,
        "answer": answer3a
    }
    logs.append(subtask_desc3a)
    print("Step 3a: ", sub_tasks[-1])
    cot_instruction_3b = "Sub-task 3b: Rank the issues based on the gathered evidence considering frequency, subtlety (difficulty to detect), and impact on downstream analysis, using a weighted decision matrix to prioritize the most problematic issues, based on outputs from Sub-task 3a." 
    cot_agent_3b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3b = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3a, answer3a]
    subtask_desc3b = {
        "subtask_id": "subtask_3b",
        "instruction": cot_instruction_3b,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3a", "answer of subtask 3a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_instruction_3b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, rank issues using weighted decision matrix, thinking: {thinking3b.content}; answer: {answer3b.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3b([taskInfo, thinking3b, answer3b], "please review the ranking of issues and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3b.extend([thinking3b, answer3b, feedback])
        thinking3b, answer3b = await cot_agent_3b(cot_inputs_3b, cot_instruction_3b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3b.id}, refining ranking of issues, thinking: {thinking3b.content}; answer: {answer3b.content}")
    sub_tasks.append(f"Sub-task 3b output: thinking - {thinking3b.content}; answer - {answer3b.content}")
    subtask_desc3b['response'] = {
        "thinking": thinking3b,
        "answer": answer3b
    }
    logs.append(subtask_desc3b)
    print("Step 3b: ", sub_tasks[-1])
    debate_instruction_4 = "Sub-task 4: Compare the ranked issues against the provided answer choices (3 and 4; 2 and 3; all of the above; 2, 3 and 4) and select the most accurate option reflecting the most common sources of difficult-to-spot erroneous results, justifying the choice with evidence and reasoning, based on outputs from Sub-task 3b." 
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
            agents.append(f"Debate agent {agent.id}, round {r}, selecting most accurate answer choice, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on the most common sources of difficult-to-spot erroneous results in genomics data analysis.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final answer choice, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    critic_final_agent = LLMAgentBase(["feedback", "correct"], "Final Critic Agent", model=self.node_model, temperature=0.0)
    N_critic = self.max_round
    final_inputs = [taskInfo, thinking4, answer4]
    for i in range(N_critic):
        feedback, correct = await critic_final_agent(final_inputs, "Please critically evaluate the final selected answer choice for any overlooked domain counterexamples or over-counted issues.", i, is_sub_task=True)
        agents.append(f"Final Critic agent, round {i}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        final_inputs.extend([feedback])
        thinking4, answer4 = await final_decision_agent_4(final_inputs, "Refine final decision based on critic feedback.", i + 1, is_sub_task=True)
        agents.append(f"Final Decision agent, refining final answer, thinking: {thinking4.content}; answer: {answer4.content}")
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs